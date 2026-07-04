# Demo recording script for VHS (https://github.com/charmbracelet/vhs).
#
# Generate the GIF:
#   1. install vhs        (macOS: brew install vhs · or see the repo)
#   2. activate the venv  (so `rag-eval` is on PATH) and set GROQ_API_KEY
#   3. run:  vhs docs/demo.tape   (or: make demo)
#
# Produces docs/demo.gif — doctor → cited answer → honest decline → config table.

Output docs/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1200
Set Height 780
Set Padding 18
Set TypingSpeed 35ms
Set Theme "Dracula"

# Preflight checklist.
Type "rag-eval doctor"
Enter
Sleep 4s

# A question the corpus can answer — note the resolved citation.
Type "rag-eval ask 'How do I create a branch and switch to it?'"
Enter
Sleep 10s

# A question it CANNOT answer — the system declines instead of hallucinating.
Type "rag-eval ask 'How do I configure Git to sign commits with an SSH key?'"
Enter
Sleep 10s

# The config-comparison experiment (retrieval-only, no LLM calls).
Type "rag-eval sweep"
Enter
Sleep 18s

Sleep 2s
