import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import math

Pdata = np.array([[2.0,5],[3,4],[4,2],[5,1]])   # data of class P
Ndata = np.array([[6.0,10],[7,5],[8,6],[9,8]])  # data of class N

minLim = 0
maxLim = 10
kernel = 'rbf' # kernel of SVM
removeRadius = .1
h = .02 # mesh step size
boundary = 0 # saves contour of SVM's boundary, initially zero

# click listener, adds or removes data points
def onclick(event):
    if event.dblclick:
        add_point('p' if event.button == 1 else 'n', event.xdata, event.ydata)
    else:
        remove_point(event.xdata, event.ydata)


# add point to class depending on label
def add_point(label, x, y):
    global Pdata, Ndata
    if label == 'p':
        Pdata = np.concatenate((Pdata, [[x, y]])) #...
    else:
        Ndata = np.concatenate((Ndata, [[x, y]]))

    update()


# remove point which is in radius of mouse click
def remove_point(x, y):
    global Pdata, Ndata

    # search for nearest point and delete from Pdata/Ndata
    index = -1
    minDist = 10e4

    i = 0
    for p in Pdata:
        dist = math.sqrt((x-p[0])*(x-p[0]) + (y-p[1])*(y-p[1]))
        if dist < minDist:
            inData = 'p'
            minDist = dist
            if dist < removeRadius:
                index = i
        i = i +1

    i = 0
    for n in Ndata:
        dist = math.sqrt((x-n[0])*(x-n[0]) + (y-n[1])*(y-n[1]))
        if dist < minDist:
            inData = 'n'
            minDist = dist
            if dist < removeRadius:
                index = i
        i = i +1

    if not index == -1:
        if inData == 'p' and len(Pdata) > 1:
            Pdata = np.delete(Pdata, index, axis=0)
        elif inData == 'n' and len(Ndata) > 1:
            Ndata = np.delete(Ndata, index, axis=0)


    update()



# retraint SVM and redraw plot and mesh (boundary)
def update():
    global scSV, scP, scN, fig, clf

    # retrain classifier
    x_train = np.append(Pdata, Ndata, axis=0)
    y_train = np.append(np.zeros(len(Pdata), dtype=int), np.ones(len(Ndata), dtype=int), axis=0)

    clf.fit(x_train, y_train) 

    # update scatter plots
    p = np.hsplit(Pdata, 2)
    n = np.hsplit(Ndata, 2)

    scSV.set_offsets(np.c_[clf.support_vectors_[:, 0], clf.support_vectors_[:, 1]])
    scP.set_offsets(np.c_[p[0],p[1]])
    scN.set_offsets(np.c_[n[0], n[1]])
    fig.canvas.draw_idle()

    draw_decision_boundary()

    plt.pause(0.0001)

# draw mesh of two colors: red for class P, blue for class N
def draw_decision_boundary():
    global boundary, Pdata, Ndata, ax, clf

    # remove old boundary if exists
    if not boundary == 0:
        for b in boundary.collections:
            b.remove()

    # create a mesh to plot in
    xx, yy = np.meshgrid(np.arange(minLim, maxLim, h),
                         np.arange(minLim, maxLim, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    boundary = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)



# initial classifier training
clf = svm.SVC(kernel=kernel)

x_train = np.append(Pdata, Ndata, axis=0)
y_train = np.append(np.zeros(len(Pdata), dtype=int), np.ones(len(Ndata), dtype=int), axis=0)

clf.fit(x_train, y_train) 


# draw plot
p = np.hsplit(Pdata, 2)
n = np.hsplit(Ndata, 2)

# enable interactive mode to update scatters
plt.ion()
fig, ax = plt.subplots()
x, y = [],[]

# highlight support vectors
scSV = ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, zorder=10, edgecolors="k", facecolors="w")
# scatter plot for class P
scP = ax.scatter(p[0],p[1], color='b', s=40, zorder=50)
# scatter plot for class N
scN = ax.scatter(n[0],n[1], color='r', s=40, zorder=50)

plt.xlim(minLim, maxLim)
plt.ylim(minLim, maxLim)


draw_decision_boundary()

plt.connect('button_press_event', onclick) # click listener
plt.show(block=True)    # prevent window from closing immediately
