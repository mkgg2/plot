from PIL import Image
from scipy.sparse import coo_matrix
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
img=Image.open('image.jpg')

r,g,b=img.split()
rr=np.array(r)
gg=np.array(g)
bb=np.array(b)

col1=(rr>180)&(gg<180)&(bb<180)
col1=col1.astype(np.int)
cc1=coo_matrix(col1)

col2=(rr<150)&(gg>rr)&(bb<150)&(gg>bb)
col2=col2.astype(np.int)
cc2=coo_matrix(col2)

col3=(rr>170)&(gg>170)&(bb>170)
col3=col3.astype(np.int)
cc3=coo_matrix(col3)

fig=plt.figure()
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133,projection='3d')

ax1.imshow(img)
ax2.imshow(img)

ax2.scatter(cc2.col,cc2.row,s=0.1,c='green')
ax2.scatter(cc3.col,cc3.row,s=0.1,c='grey')
ax2.scatter(cc1.col,cc1.row,s=0.1,c='orange')

ax3.scatter(cc1.col,cc1.row,30,s=0.1,c='orange')
ax3.scatter(cc2.col,cc2.row,20,s=0.1,c='green')
ax3.scatter(cc3.col,cc3.row,10,s=0.1,c='grey')

plt.show()
