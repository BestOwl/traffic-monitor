#How to run tlt-converter
#https://forums.developer.nvidia.com/t/how-to-run-tlt-converter/147299

tlt-converter resnet18_trafficcamnet_pruned.etlt \
    -k tlt_encode \
    -c trafficnet_int8.txt \
    -o output_cov/Sigmoid,output_bbox/BiasAdd \
    -d 3,544,960 \
    -i nchw \
    -e trafficnet_int8.engine \
    -m 1 -t int8 -b 1
