import runpy, sys
pylib = sys.argv[1]
sys.path.insert(0, pylib)
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
