# Project Context

## Purpose
识别图片中的单个字母和数字字符 (Single character recognition for alphanumeric).

## Tech Stack
- Python 3.10+
- OpenCV (headless)
- NumPy
- Pytest (Testing)
- PyTorch (Deep Learning Framework)
- Torchvision (Models & Transformations)

## Project Conventions

### Code Style
- Follow PEP 8
- Formatter: Black
- Linter: Flake8 / Pylint

### Architecture Patterns
- Modular script structure (initially)
- Potential for CLI interface

### Testing Strategy
- Unit tests using Pytest
- Test data: Sample images of single characters

### Git Workflow
- Create feature branches for changes
- Pull Requests with review
- Archive OpenSpec changes upon completion

## Domain Context
- Input: Image file containing one character
- Output: Prediction (Char) + Confidence
- Challenge: Distinguishing similar characters (0/O, 1/I/l)

## Important Constraints
- Performance: Quick inference
- Accuracy: Prioritize correct identification of ambiguous characters
- Local execution (no cloud dependence initially)

## External Dependencies
- None currently
