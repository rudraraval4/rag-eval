# Branching

A branch in Git is a lightweight, movable pointer to a commit. Creating a branch
just writes a new pointer; it does not copy any files, which is why branching is
cheap and fast.

## Creating and switching branches

`git branch <name>` creates a branch but does not switch to it. `git switch
<name>` moves to an existing branch, and `git switch -c <name>` creates a new
branch and switches to it in one step. The older `git checkout -b <name>` does
the same thing.

## HEAD

`HEAD` is a reference to the current commit, usually by pointing at the current
branch. When you commit, the current branch pointer and `HEAD` move forward to
the new commit. A "detached HEAD" happens when `HEAD` points directly at a
commit instead of a branch.

## Deleting branches

`git branch -d <name>` deletes a branch that has already been merged. Use
`git branch -D <name>` to force-delete a branch whose changes are not merged.
