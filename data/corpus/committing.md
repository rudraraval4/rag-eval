# Committing Changes

A commit is a snapshot of the staged content plus metadata: author, committer,
timestamp, a message, and one or more parent commits. Each commit is identified
by a SHA-1 (or SHA-256) hash computed from its contents.

## Staging and committing

Use `git add <path>` to stage specific files, or `git add -A` to stage all
changes including deletions. Then `git commit -m "message"` records the staged
snapshot. `git commit -am "message"` stages already-tracked, modified files and
commits them in one step, but it does not include new untracked files.

## Amending the last commit

`git commit --amend` replaces the most recent commit with a new one that
includes any newly staged changes and lets you edit the message. Because it
rewrites the commit, avoid amending commits you have already pushed and shared.

## Good commit messages

Write a short imperative summary line (for example, "Add retry logic"), keep it
under about fifty characters, and add a blank line before any longer body. The
body should explain why the change was made, not just what changed.
