import pickle

def pkl_save(path,obj):
  with open(path, 'wb') as handle:
      pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
def pkl_load(path):
  with open(path, 'rb') as handle:
      return pickle.load(handle)