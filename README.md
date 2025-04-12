# MPI Chess Engine with GUI

This project is a distributed chess engine that uses MPI (Message Passing Interface) for parallel computation to distribute the workload of evaluating chess positions among multiple processes.

## Requirements

- Python 3.6+
- mpi4py
- python-chess
- pygame

## Installation

1. Install the required packages:

```bash
pip install mpi4py python-chess pygame
```

2. Make sure you have a working MPI implementation (like OpenMPI or MPICH) installed on your system.

## Running the Chess Game

### With GUI

To run the chess game with the GUI:

```bash
mpirun -n <number_of_processes> python chess_gui.py 
```

Where `<number_of_processes>` is the number of processes you want to use (recommended at least 2: 1 for the GUI/master and 1+ for workers).

For example:

```bash
mpirun -n 4 python chess_gui.py 
```

This will start the game with 1 master process (which also runs the GUI) and 3 worker processes.

### Console Mode

To run the original console-based version:

```bash
mpirun -n <number_of_processes> python run.py
```

## GUI Controls

- Click on a piece to select it
- Click on a valid destination square to move the selected piece
- Press **N** to start a new game
- Press **U** to undo the last two moves (your move and the AI's response)

## Folder Structure

- `run.py` - Entry point for the application
- `chess_gui.py` - GUI implementation using Pygame
- `master.py` - Master process implementation for distributed computation
- `worker.py` - Worker process implementation
- `chess_engine.py` - Chess AI and evaluation functions
- `utils.py` - Utility functions for serialization and logging

## Chess Pieces Images

For the best experience, create a `chess_pieces` folder in the same directory as the scripts and add PNG images for each chess piece with the following naming convention:

- White pieces: `wp.png`, `wr.png`, `wn.png`, `wb.png`, `wq.png`, `wk.png`
- Black pieces: `bp.png`, `br.png`, `bn.png`, `bb.png`, `bq.png`, `bk.png`

If images are not found, the application will fall back to displaying piece symbols.