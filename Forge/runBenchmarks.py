import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Forge.benchmark import run_all_benchmarks

run_all_benchmarks()