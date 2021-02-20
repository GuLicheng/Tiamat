# GCN for Saliency Object Detection  
## input  
Sometimes the input in Saliency Object Detection is a image with RGB-channels, or even depth-channels
### RGB
RGB image usually consist of 3-channels, we often reshape the origin image to our respected size(such as 300 * 300), and the output of our data loader may be a 4 dimensions tensor([B, C, H, W])  
B: batch size  
C: channels, for rgb usually be 3  
H, W: size of image, it can be see as a H*W vector

### depth
I don't know

## propagation  
for graph convolution layers, the standard input is consist of feature matrix and adjacent matrix, for   
**H<sub>i+1</sub> = $\sigma$(AH<sub>i</sub>W)**  
H<sub>i+1</sub> is output of GCN layer, A is adjacent matrix of graph and H<sub>i</sub> is input feature, W is a learnable weight matrix which can be used to change channels(features dimensions) of input.

# GCN 
GCN can be viewed as two steps: select nodes and build edges
## Select Nodes
node can be a feature vector, in SOD, it may be a channel or a feature map or a super pixel
### channel/feature
simply view a channel as a feature, it may same as traditional method
### super pixel
use some algorithm for reshape, or split pixel block and view each block as a node
## Build Edges
Usually built by learning

# Propagation Detail



