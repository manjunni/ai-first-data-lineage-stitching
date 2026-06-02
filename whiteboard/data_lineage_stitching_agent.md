# AI Agent 05 – Data Lineage Stitching Agent

## Agentic AI Solutions for Data Trust  
Capgemini Insights & Data – December 2025

---

## Purpose of this Deck
This document describes the functionality of the **Data Lineage Stitching Agent**, part of a broader suite of AI agents supporting data governance and trust.

---

## Problem Statement
Most data catalog tools can parse lineage **within a single system**. However, they struggle to connect lineage across systems (e.g., S3 → Spark → Snowflake → BI tools).

### The Agent should:
- Collect lineage fragments from multiple sources  
- Normalize context across systems  
- Stitch fragments into a unified lineage graph  
- Generate visual cross-system lineage maps  
- Validate with SMEs and publish to the catalog  

---

## Overall Objective
Enable a human-in-the-loop AI agent to extract, complete, and stitch end-to-end data lineage and source-to-target mappings across multiple systems, producing validated mappings and clear visuals that can be published to the catalog—with confidence scores, audit trail, and SME approvals.

---

## How It Works (Simple View)
1. Read documentation (SQL, ETL, BI)
2. Build small lineage fragments
3. Connect fragments across systems
4. Validate with SMEs
5. Generate visuals and publish to catalog

---

## High-Level Process

### Step 1: Define Standards
- Define valid lineage mapping rules  
- Set naming conventions and IDs  
- Establish confidence scoring  

### Step 2: Curate Training Examples
- Create “gold set” lineage examples  
- Include various patterns (SQL, dbt, BI, ETL)  

### Step 3: Gather Artifacts
- SQL scripts, ETL configs, BI models  
- Data contracts, dictionaries, policies  

---

## Core Agent Workflow

### Step 4: Parse Lineage 
- Extract sources, targets, transformations  
- Generate structured JSON  

### Step 5: Capture SME Knowledge
- Conduct SME interviews  
- Build ownership registry  

### Step 6: Generate Draft Mappings
- Combine parsed fragments and metadata  
- Produce initial mappings  

### Step 7–8: Gap Detection & SME Interaction
- Detect missing mappings  
- Generate SME questions  
- Collect responses  

### Step 9: Finalize Mappings
- Apply SME inputs  
- Complete mappings  

### Step 10: Stitch Lineage
- Connect fragments into full lineage graph  

### Step 11: Generate Visuals
- Create diagrams (Mermaid, Graphviz)  

### Step 12: Publish to Catalog 
- Load mappings and lineage graphs  

---

## Key Capabilities
- Cross-system lineage stitching  
- Human-in-the-loop validation  
- Confidence-based scoring  
- Full audit trail  
- Automated catalog publishing  

---

## Summary
This agent enables organizations to move from fragmented lineage to complete, trusted, end-to-end lineage using AI and human validation.
