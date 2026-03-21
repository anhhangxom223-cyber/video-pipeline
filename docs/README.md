# Documentation Index

This folder contains architecture and freeze specifications for the video/audio pipeline.

---

## 📦 Structure

### Architecture
- docs/architecture/
  - Video pipeline architecture
  - Module map
  - Execution flow

### Freeze Specifications
- docs/freeze/
  - Video pipeline freeze (v3.2.0)
  - Audio pipeline freeze (v4.2.1)

---

## 🚨 Critical Rules (MUST FOLLOW)

### 1. Architecture Freeze
- No new threads, queues, or async primitives
- Video pipeline: Reader → Processor → Encoder only
- Audio pipeline runs AFTER video completes

### 2. Determinism
- Same input MUST produce same output
- No randomness
- No time-dependent logic

### 3. DTO Contracts
- All DTOs are immutable (frozen=True)
- No field changes allowed

### 4. Interface Freeze
- Method signatures are locked
- No parameter changes

### 5. Dependency Rules
- No new external dependencies without approval
- No circular imports

---

## 🧪 Purpose

These documents act as:
- Source of truth for system design
- Constraints for development
- Validation rules for testing

---

## ⚠️ For AI / Code Review Tools

When analyzing this repository:
- Treat freeze documents as STRICT constraints
- Report any violation of:
  - architecture
  - interfaces
  - DTO immutability
  - dependency rules
