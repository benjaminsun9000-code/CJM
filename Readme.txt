This is project of generating Canonical Jordan Form of a matrix based on Fillipov's proof.

Fillipov's proof was mentioned from Strange's Linear Algebra book and studied in detail with AI. 
This proof is based on mathematical induction and naturally leads to recursive implementation. 

Project structure:
-- RREF related:
  -- find Null space basis for an input matrix
  -- find the indices of pivot point in a matrix of RREF form
  -- given some basis of a space, expand this given basis to the whole space defined by the column space of an input matrix
  -- solve Ax = b. Return one solution (not necessarily the projection of b in A) if there is solution, or return None.
-- Using Gram-Schmit to return a orthornormal basis given an input matrix
-- Finding the dorminant eigen value and associated eigen vectors of a matrix if posssible
-- Given a linear transformation T, find its restricted transformation in a subspace defined by an input matrix columns
-- Recursively find CJM based on Fillipov's proof 
