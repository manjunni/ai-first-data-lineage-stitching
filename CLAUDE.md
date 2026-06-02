# [Programme Name] -- AI-First Architecture

This project follows the AI-First Business Service Architecture methodology.

## Methodology Source

<!-- This import gives Claude Code full awareness of the methodology toolkit -->
<!-- All references are bundled in .claude/agents/references/ -- no external repository required -->

@.claude/agents/TOOLKIT-REFERENCE.md

## Programme Context

<!-- Fill in your programme-specific details below -->

**Programme Name:** AI-First Architecture
**Organisation:** Capgemini
**Programme Mission:** To create accurate, complete, and catalog-ready data lineage and mappings across systems using an AI agent supported by human expertise.
**Key Stakeholders:** Ralf Teschner, Greg Solomon

## Architecture Outputs

All architecture artifacts for this programme are stored in:
- `architecture/canonical-model.json` -- Single source of truth (canonical service domain model)
- `architecture/diagrams/` -- Generated SVG architecture diagrams
- `architecture/personas/` -- AI Persona definition documents
- `architecture/measurements/` -- 5-tier measurement blueprints
- `architecture/governance/` -- Readiness scorecards, dependency register, compliance reports

## Working Conventions

- The canonical model JSON is the spine -- all diagrams and reports derive from it
- Use the `canonical-model-manager` skill to create and update the model
- Use the `service-architecture-renderer` skill to generate diagrams after model changes
- Follow the 7-phase workflow: Strategic Alignment > L0 Discovery > L1 Drill-Down > Persona Definition > Architecture Completion > Measurement Design > Compliance Review
- Each phase produces concrete artifacts before moving to the next
