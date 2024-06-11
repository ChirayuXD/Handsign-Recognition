import h5py

f = h5py.File('Model/keras_model.h5', 'r')
print(f.attrs.get('keras_version'))