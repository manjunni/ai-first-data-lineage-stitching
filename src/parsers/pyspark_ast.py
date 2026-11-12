"""
AST-based PySpark dataflow evaluator for Databricks ``.py`` notebooks.

The regex tracker in ``notebook_parser`` only follows literal table names and
single-parent ``var = var.op(...)`` chains. Real notebooks build names with
f-strings, read widgets, wrap reads/writes in helper functions and join several
DataFrames together. This module walks the Python AST and abstractly evaluates
just enough of it to recover:

* string values — literals, f-strings, ``str.format``, ``+``, ``lower/upper``,
  with unresolved parts kept as ``{placeholder}`` templates;
* DataFrame values — the set of upstream table/path names feeding them,
  merged across joins, unions and opaque helper calls;
* writes — ``saveAsTable / insertInto / save / toTable / writeTo``.

Placeholders use the ``{name}`` / ``{name:upper}`` / ``{name:lower}`` syntax.
The stitcher later pattern-matches templated names against concrete tables.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Abstract values
# ---------------------------------------------------------------------------

_PH = re.compile(r"\{([A-Za-z_]\w*)(?::(upper|lower))?\}")


@dataclass(frozen=True)
class Str:
    text: str
    soft: bool = False  # depends on a runtime-overridable default (widget)

    @property
    def templated(self) -> bool:
        return bool(_PH.search(self.text))


@dataclass
class DF:
    sources: list[str] = field(default_factory=list)
    ops: list[str] = field(default_factory=list)
    opaque: list[str] = field(default_factory=list)
    soft: bool = False

    def merged(self, *others: "DF", op: str | None = None) -> "DF":
        out = DF(list(self.sources), list(self.ops), list(self.opaque), self.soft)
        for o in others:
            for s in o.sources:
                if s not in out.sources:
                    out.sources.append(s)
            out.ops.extend(o.ops)
            out.opaque.extend(o.opaque)
            out.soft = out.soft or o.soft
        if op:
            out.ops.append(op)
        return out


@dataclass
class ListVal:
    items: list


class _Spark: ...
class _Dbutils: ...


@dataclass
class _Reader:
    pass


@dataclass
class _Writer:
    df: DF
    target: Str | None = None


class _Unknown: ...


UNKNOWN = _Unknown()

_DF_OPS = {
    "filter", "where", "join", "crossJoin", "groupBy", "agg", "withColumn",
    "withColumnRenamed", "union", "unionAll", "unionByName", "distinct",
    "dropDuplicates", "sort", "orderBy",
}
_READ_TERMINALS = {"load", "table", "parquet", "csv", "json", "orc", "text", "delta"}
_WRITE_TERMINALS = {"saveAsTable", "insertInto", "save", "toTable", "start",
                    "parquet", "csv", "json", "orc", "text"}
_WRITETO_TERMINALS = {"append", "create", "createOrReplace", "replace",
                      "overwrite", "overwritePartitions"}


@dataclass
class WriteEvent:
    target: Str
    df: DF


@dataclass
class EvalResult:
    writes: list[WriteEvent] = field(default_factory=list)
    sql_texts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def _case(s: Str, mode: str) -> Str:
    out, last = [], 0
    for m in _PH.finditer(s.text):
        lit = s.text[last:m.start()]
        out.append(lit.upper() if mode == "upper" else lit.lower())
        out.append("{%s:%s}" % (m.group(1), mode))
        last = m.end()
    tail = s.text[last:]
    out.append(tail.upper() if mode == "upper" else tail.lower())
    return Str("".join(out), s.soft)


def _concat(*parts: Str) -> Str:
    return Str("".join(p.text for p in parts), any(p.soft for p in parts))


def _expr_hint(node: ast.AST) -> str:
    """Name used for a placeholder when an expression can't be resolved."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _expr_hint(node.func)
    if isinstance(node, ast.Subscript):
        return _expr_hint(node.value)
    return "expr"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class _Evaluator:
    MAX_DEPTH = 6
    MAX_LOOP = 25

    def __init__(self) -> None:
        self.result = EvalResult()
        self.widgets: dict[str, str] = {}
        self.functions: dict[str, ast.FunctionDef] = {}
        self.depth = 0

    # -- statements --------------------------------------------------------

    def run(self, stmts: list[ast.stmt], env: dict) -> object | None:
        """Execute statements; return the value of a ``return`` if hit."""
        ret = None
        for st in stmts:
            r = self.stmt(st, env)
            if r is not None:
                ret = r
        return ret

    def stmt(self, st: ast.stmt, env: dict):
        if isinstance(st, ast.FunctionDef):
            self.functions[st.name] = st
        elif isinstance(st, ast.Assign):
            val = self.eval(st.value, env)
            for tgt in st.targets:
                self.bind(tgt, val, env)
        elif isinstance(st, ast.AnnAssign) and st.value is not None:
            self.bind(st.target, self.eval(st.value, env), env)
        elif isinstance(st, ast.AugAssign):
            cur = self.eval(st.target, env) if isinstance(st.target, ast.Name) else UNKNOWN
            rhs = self.eval(st.value, env)
            if isinstance(cur, Str) and isinstance(rhs, Str) and isinstance(st.op, ast.Add):
                self.bind(st.target, _concat(cur, rhs), env)
        elif isinstance(st, ast.Expr):
            self.eval(st.value, env)
        elif isinstance(st, ast.Return):
            return self.eval(st.value, env) if st.value is not None else UNKNOWN
        elif isinstance(st, ast.For):
            it = self.eval(st.iter, env)
            items = it.items[: self.MAX_LOOP] if isinstance(it, ListVal) and it.items else [None]
            ret = None
            for item in items:
                if item is None:
                    item = Str("{%s}" % _expr_hint(st.target))
                self.bind(st.target, item, env)
                r = self.run(st.body, env)
                ret = r if r is not None else ret
            r = self.run(st.orelse, env)
            return r if r is not None else ret
        elif isinstance(st, (ast.If, ast.While)):
            r1 = self.run(st.body, env)
            r2 = self.run(st.orelse, env)
            return r1 if r1 is not None else r2
        elif isinstance(st, ast.With):
            for item in st.items:
                if item.optional_vars is not None:
                    self.bind(item.optional_vars, self.eval(item.context_expr, env), env)
                else:
                    self.eval(item.context_expr, env)
            return self.run(st.body, env)
        elif isinstance(st, ast.Try):
            r = self.run(st.body, env)
            for h in st.handlers:
                self.run(h.body, env)
            self.run(st.orelse, env)
            self.run(st.finalbody, env)
            return r
        return None

    def bind(self, target: ast.AST, val, env: dict) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = val
        elif isinstance(target, (ast.Tuple, ast.List)):
            items = val.items if isinstance(val, ListVal) and len(val.items) == len(target.elts) else None
            for i, elt in enumerate(target.elts):
                if items is not None:
                    self.bind(elt, items[i], env)
                elif isinstance(elt, ast.Name):
                    # Unpacking an unresolved value (e.g. run_date.split("-")):
                    # each name becomes its own placeholder.
                    soft = isinstance(val, ListVal) and any(getattr(x, "soft", False) for x in val.items)
                    env[elt.id] = Str("{%s}" % elt.id, soft)

    # -- expressions -------------------------------------------------------

    def eval(self, node: ast.AST | None, env: dict):  # noqa: C901 — dispatch table
        if node is None:
            return UNKNOWN
        try:
            return self._eval(node, env)
        except RecursionError:
            return UNKNOWN

    def _eval(self, node: ast.AST, env: dict):
        if isinstance(node, ast.Constant):
            return Str(node.value) if isinstance(node.value, str) else UNKNOWN
        if isinstance(node, ast.JoinedStr):
            return self._fstring(node, env)
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id == "spark":
                return _Spark()
            if node.id == "dbutils":
                return _Dbutils()
            return UNKNOWN
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            l, r = self.eval(node.left, env), self.eval(node.right, env)
            if isinstance(l, Str) and isinstance(r, Str):
                return _concat(l, r)
            if isinstance(l, ListVal) and isinstance(r, ListVal):
                return ListVal(l.items + r.items)
            return UNKNOWN
        if isinstance(node, (ast.List, ast.Tuple)):
            return ListVal([self.eval(e, env) for e in node.elts])
        if isinstance(node, ast.ListComp) or isinstance(node, ast.GeneratorExp):
            return self._listcomp(node, env)
        if isinstance(node, ast.Subscript):
            return self._subscript(node, env)
        if isinstance(node, ast.Attribute):
            return self._attribute(node, env)
        if isinstance(node, ast.Call):
            return self._call(node, env)
        if isinstance(node, ast.IfExp):
            a = self.eval(node.body, env)
            return a if not isinstance(a, _Unknown) else self.eval(node.orelse, env)
        return UNKNOWN

    def _fstring(self, node: ast.JoinedStr, env: dict) -> Str:
        parts: list[Str] = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(Str(str(v.value)))
            elif isinstance(v, ast.FormattedValue):
                parts.append(self.as_str(v.value, env))
        return _concat(*parts) if parts else Str("")

    def as_str(self, node: ast.AST, env: dict) -> Str:
        val = self.eval(node, env)
        if isinstance(val, Str):
            return val
        return Str("{%s}" % _expr_hint(node))

    def _listcomp(self, node, env: dict):
        gen = node.generators[0]
        it = self.eval(gen.iter, env)
        local = dict(env)
        out = []
        if isinstance(it, ListVal) and it.items:
            for item in it.items[: self.MAX_LOOP]:
                self.bind(gen.target, item, local)
                out.append(self.eval(node.elt, local))
        else:
            soft = getattr(it, "soft", False)
            if isinstance(gen.target, ast.Name):
                local[gen.target.id] = Str("{%s}" % gen.target.id, soft)
            out.append(self.eval(node.elt, local))
        return ListVal(out)

    def _subscript(self, node: ast.Subscript, env: dict):
        base = self.eval(node.value, env)
        if isinstance(base, ListVal):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, int):
                try:
                    return base.items[sl.value]
                except IndexError:
                    return UNKNOWN
            if isinstance(sl, ast.Slice):
                lo = sl.lower.value if isinstance(sl.lower, ast.Constant) else None
                hi = sl.upper.value if isinstance(sl.upper, ast.Constant) else None
                return ListVal(base.items[lo:hi])
            return _merge_list(base)
        if isinstance(base, DF):
            return UNKNOWN  # column reference df["col"]
        return UNKNOWN

    def _attribute(self, node: ast.Attribute, env: dict):
        base = self.eval(node.value, env)
        attr = node.attr
        if isinstance(base, _Spark):
            if attr in ("read", "readStream"):
                return _Reader()
            return base
        if isinstance(base, _Dbutils):
            return base
        if isinstance(base, _Reader):
            return base
        if isinstance(base, DF):
            if attr in ("write", "writeStream"):
                return _Writer(base)
            if attr in ("na", "stat"):
                return base
            return UNKNOWN  # column access df.col
        if isinstance(base, _Writer):
            return base
        return UNKNOWN

    def _call(self, node: ast.Call, env: dict):
        func = node.func

        # Plain function call: user-defined helper or opaque library call
        if isinstance(func, ast.Name):
            name = func.id
            if name in self.functions and self.depth < self.MAX_DEPTH:
                return self._call_user(self.functions[name], node, env)
            if name in ("str",):
                return self.as_str(node.args[0], env) if node.args else UNKNOWN
            if name in ("list", "tuple", "sorted") and node.args:
                return self.eval(node.args[0], env)
            dfs = self._df_args(node, env)
            if dfs:
                return dfs[0].merged(*dfs[1:]).merged(DF(opaque=[name]))
            return UNKNOWN

        if not isinstance(func, ast.Attribute):
            return UNKNOWN

        method = func.attr

        # DeltaTable.forName(spark, "x") / forPath(spark, "x")
        if isinstance(func.value, ast.Name) and func.value.id == "DeltaTable" and method in ("forName", "forPath"):
            arg = node.args[-1] if node.args else None
            return DF([self.as_str(arg, env).text]) if arg is not None else UNKNOWN

        recv = self.eval(func.value, env)

        # dbutils.widgets.*
        if isinstance(recv, _Dbutils):
            if method in ("text", "dropdown", "combobox", "multiselect") and len(node.args) >= 2:
                n, d = self.eval(node.args[0], env), self.eval(node.args[1], env)
                if isinstance(n, Str) and isinstance(d, Str):
                    self.widgets[n.text] = d.text
                return UNKNOWN
            if method == "get" and node.args:
                n = self.eval(node.args[0], env)
                if isinstance(n, Str):
                    default = self.widgets.get(n.text, "")
                    if default:
                        return Str(default, soft=True)
                    return Str("{%s}" % n.text)
            return UNKNOWN

        # spark.table / spark.sql
        if isinstance(recv, _Spark):
            if method == "table" and node.args:
                s = self.as_str(node.args[0], env)
                return DF([s.text], soft=s.soft)
            if method == "sql" and node.args:
                s = self.eval(node.args[0], env)
                if isinstance(s, Str) and not s.templated:
                    self.result.sql_texts.append(s.text)
                    return DF(_sql_select_sources(s.text))
                return DF()
            return UNKNOWN

        # spark.read... / spark.readStream...
        if isinstance(recv, _Reader):
            if method in _READ_TERMINALS:
                if node.args:
                    s = self.as_str(node.args[0], env)
                    return DF([s.text], soft=s.soft)
                path_kw = next((k.value for k in node.keywords if k.arg == "path"), None)
                if path_kw is not None:
                    s = self.as_str(path_kw, env)
                    return DF([s.text], soft=s.soft)
                return DF()
            return recv  # format/option/schema/...

        # df.write... / df.writeStream... / df.writeTo(x)...
        if isinstance(recv, _Writer):
            if recv.target is not None and method in _WRITETO_TERMINALS:
                self.result.writes.append(WriteEvent(recv.target, recv.df))
                return UNKNOWN
            if method in _WRITE_TERMINALS:
                tgt = node.args[0] if node.args else next(
                    (k.value for k in node.keywords if k.arg in ("name", "path", "tableName")), None)
                if tgt is not None:
                    self.result.writes.append(WriteEvent(self.as_str(tgt, env), recv.df))
                return UNKNOWN
            return recv

        if isinstance(recv, DF):
            if method == "writeTo" and node.args:
                return _Writer(recv, self.as_str(node.args[0], env))
            others = self._df_args(node, env)
            op = method if method in _DF_OPS else None
            out = recv.merged(*others, op=op)
            return out

        # String methods
        if isinstance(recv, Str):
            if method in ("upper", "lower"):
                return _case(recv, method)
            if method in ("strip", "lstrip", "rstrip"):
                return recv
            if method == "split":
                sep = self.eval(node.args[0], env) if node.args else Str(" ")
                if not recv.templated and isinstance(sep, Str):
                    return ListVal([Str(p, recv.soft) for p in recv.text.split(sep.text)])
                return ListVal([recv])  # unresolved; unpacking yields placeholders
            if method == "format":
                return _format(recv, {k.arg: self.as_str(k.value, env) for k in node.keywords if k.arg},
                               [self.as_str(a, env) for a in node.args])
            if method == "replace" and len(node.args) == 2:
                a, b = self.eval(node.args[0], env), self.eval(node.args[1], env)
                if isinstance(a, Str) and isinstance(b, Str):
                    return Str(recv.text.replace(a.text, b.text), recv.soft)
            return UNKNOWN

        # Method on an unknown object (e.g. F.col(...), strip_vat module func)
        dfs = self._df_args(node, env)
        if dfs:
            label = f"{_expr_hint(func.value)}.{method}"
            return dfs[0].merged(*dfs[1:]).merged(DF(opaque=[label]))
        return UNKNOWN

    def _df_args(self, node: ast.Call, env: dict) -> list[DF]:
        out = []
        for a in list(node.args) + [k.value for k in node.keywords]:
            v = self.eval(a, env)
            if isinstance(v, DF):
                out.append(v)
            elif isinstance(v, ListVal):
                m = _merge_list(v)
                if isinstance(m, DF):
                    out.append(m)
        return out

    def _call_user(self, fn: ast.FunctionDef, node: ast.Call, env: dict):
        local = dict(env)
        params = [a.arg for a in fn.args.args]
        defaults = fn.args.defaults
        for name, d in zip(params[len(params) - len(defaults):], defaults):
            local[name] = self.eval(d, env)
        for name, a in zip(params, node.args):
            local[name] = self.eval(a, env)
        for k in node.keywords:
            if k.arg:
                local[k.arg] = self.eval(k.value, env)
        self.depth += 1
        try:
            ret = self.run(fn.body, local)
        finally:
            self.depth -= 1
        return ret if ret is not None else UNKNOWN


def _merge_list(lv: ListVal):
    dfs = [i for i in lv.items if isinstance(i, DF)]
    if not dfs:
        return UNKNOWN
    return dfs[0].merged(*dfs[1:])


def _format(s: Str, kwargs: dict[str, Str], args: list[Str]) -> Str:
    soft = s.soft or any(v.soft for v in kwargs.values()) or any(a.soft for a in args)
    pos = iter(args)

    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key in kwargs:
            return kwargs[key].text
        return m.group(0)

    text = _PH.sub(repl, s.text)
    if args:
        text = re.sub(r"\{\}", lambda _m: next(pos, Str("{}")).text, text)
    return Str(text, soft)


def _sql_select_sources(sql: str) -> list[str]:
    """Tables read by a plain SELECT passed to spark.sql (writes are handled by sqlglot elsewhere)."""
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(sql, read="spark")
    except Exception:
        return []
    if not isinstance(tree, exp.Select):
        return []
    out = []
    for t in tree.find_all(exp.Table):
        name = ".".join(p for p in (t.catalog, t.db, t.name) if p)
        if name and name not in out:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_python(source: str) -> EvalResult:
    """Parse ``source`` and return the writes (with upstream DataFrame sources) it performs."""
    tree = ast.parse(_strip_magics(source))
    ev = _Evaluator()
    ev.run(tree.body, {})
    return ev.result


def _strip_magics(source: str) -> str:
    """Drop Databricks/IPython magic lines (``%pip``, ``!ls``) that aren't valid Python."""
    return "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )


_NOTEBOOK_PATH = re.compile(r"^#\s*Notebook path:\s*(\S+)", re.MULTILINE | re.IGNORECASE)


def notebook_path_from_header(source: str) -> str | None:
    m = _NOTEBOOK_PATH.search(source)
    return m.group(1).rstrip("/") if m else None


def placeholder_names(text: str) -> list[str]:
    return [m.group(1) for m in _PH.finditer(text)]


PLACEHOLDER_RE = _PH
