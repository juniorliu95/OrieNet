# OrieNet
This work is the DRNN part of OrieNet.

platform:tensorflow

#files:
fingerprit.py: structure of the network
Prepare_Data.py: making tfrecord files of the network
test.py: output test images' orientation fields
test_FingerPrint1.py:Training process of the network
m_files/getdecmatrix.m show the visualized result.

#running demo:
run 'test.py' to get orientation fields.
run m_files/getdecmatrix.m to get the visualized result.


#training
first convert your database to tfrecord files.
labels are of shape [20,20,3].
images are of [160,160], background masked.

then run test_FingerPrint1.py to train your own database.
the next steps are the same with 'running demo'.
