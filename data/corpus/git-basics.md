# Git Basics

Git is a distributed version control system. Every clone of a repository is a
full copy, including the entire history, so most operations are local and fast.

## Initializing a repository

`git init` creates a new repository in the current directory. It adds a hidden
`.git` folder that stores all history, configuration, and objects. The directory
you run it in becomes the working tree.

## The three areas

Git tracks content across three areas:

- The **working tree** is the set of files you edit.
- The **staging area** (also called the index) holds the changes you have marked
  to include in the next commit.
- The **repository** is the committed history stored under `.git`.

You move changes from the working tree to the staging area with `git add`, and
from the staging area into the repository with `git commit`.

## Checking status

`git status` shows which files are modified, which are staged, and which are
untracked. It is the fastest way to see the state of the three areas.
