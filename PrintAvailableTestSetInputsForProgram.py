from Common import GetTestSet
from Common import Constants

set = GetTestSet()
for featureVector in set:
    print(featureVector[Constants.Database] + ' ' + featureVector[Constants.RecordNumber])