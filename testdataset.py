import sys,os
import logistic_sgd

if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit('Usage: %s dataset' % sys.argv[0])
  if not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: Dataset %s was not found!' % sys.argv[1])
  logistic_sgd.sgd_optimization(dataset=sys.argv[1])