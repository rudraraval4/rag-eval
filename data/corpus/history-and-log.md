# Inspecting History

Git records a complete history of commits that you can search and inspect.

## Viewing the log

`git log` shows commits in reverse chronological order. Useful options include
`--oneline` for a compact one-line-per-commit view, `--graph` to draw the branch
and merge structure, and `-n <count>` to limit the number of commits shown.

## Seeing what changed

`git show <commit>` displays the metadata and the diff of a single commit.
`git diff` shows unstaged changes, `git diff --staged` shows staged changes, and
`git diff <a> <b>` compares two commits or branches.

## Who changed a line

`git blame <file>` annotates each line of a file with the commit, author, and
date that last modified it. It is the standard way to find when and why a
specific line was introduced.

## Searching history

`git log -S "<string>"` (the "pickaxe") finds commits that added or removed a
given string, which is useful for tracking down when a piece of code appeared or
disappeared.
