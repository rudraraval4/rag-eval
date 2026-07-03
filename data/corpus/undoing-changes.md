# Undoing Changes

Git offers several ways to undo work, and they differ in whether they rewrite
history and whether they touch your files.

## Reset

`git reset` moves the current branch pointer and optionally changes the staging
area and working tree:

- `git reset --soft <commit>` moves the branch pointer but keeps the staging
  area and working tree, so the changes remain staged.
- `git reset --mixed <commit>` (the default) also resets the staging area but
  keeps the working tree.
- `git reset --hard <commit>` resets the staging area and the working tree,
  discarding all uncommitted changes. This is destructive.

## Revert

`git revert <commit>` creates a new commit that undoes the changes of an earlier
commit without rewriting history. Because it adds a commit rather than removing
one, it is safe to use on shared branches.

## Restoring files

`git restore <path>` discards uncommitted changes in the working tree for the
given files. `git restore --staged <path>` unstages a file while keeping its
working-tree changes.
