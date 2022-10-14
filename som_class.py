import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from PIL import Image
import os
import pyglet
from matplotlib.colors import LinearSegmentedColormap

class SOM(object):
    def __init__(self,Rows,Cols,Dim,data):
        self.lattice = (Rows,Cols,Dim)
        self.data = data
        self.listBMU = []
        self.umatrix = []
        self.LR = []
        self.NR = []
        
    def fun_trainSOM(self,LR0,lamLN,NR0,stepMax):
        print("LR0=" + str(LR0) + ",lamLN=" + str(lamLN) + ",NR0=" + str(NR0) + ",stepMax=" + str(stepMax))
        self.LR0 = LR0
        self.lamLN = lamLN
        self.NR0 = NR0

        X = np.linspace(0.05, 0.1, self.lattice[0])
        Y = np.linspace(0.25, 0.3, self.lattice[1]) 
        arr = [] 
        for x in X :
            arr1 = [] 
            for y in Y : 
                arr1.append([x,y])
            arr.append(arr1)

        self.som = np.asarray(arr)

        self.listBMU = []

        if not os.path.exists('image_dir'):
            os.makedirs('image_dir')
        image_names = []

        print("Start training")
        error_list = []

        for t in range(stepMax + 1):
            if t % int(stepMax / 20) == 0:
                print('step = ' + str(t))
                arr = []
                arr1 = []
                # if t != 0 and t % int(stepMax / 5) == 0:
                #     error_list.append([t, self.fun_quant_err()])
                for iter in self.som:
                    arr = np.hstack((arr, iter[:,0]))
                    arr1 = np.hstack((arr1, iter[:,1]))

                plt.figure(figsize=(15, 15))
                plt.plot(self.data[:,0], self.data[:,1], linestyle='', marker='*',  markersize=4, color="Red")

                for i in range(self.lattice[0]) :
                    if t == 0 or t > stepMax/10*9 :
                        plt.plot(self.som[i][:,0], self.som[i][:,1], linestyle='', marker='o',  markersize=8, color="Blue")
                    else :
                        plt.plot(self.som[i][:,0], self.som[i][:,1], linestyle='-', marker='o',  markersize=4, color="Blue")
                
                for j in range(self.lattice[1]) :
                    aar = []
                    aar1 = []
                    for i in range(self.lattice[0]) :
                        aar.append(self.som[i][j,0])
                        aar1.append(self.som[i][j,1])
                    if t == 0 or t > stepMax/10*9 :
                        plt.plot(aar, aar1, linestyle='', marker='o',  markersize=8, color="Blue")
                    else :
                        plt.plot(aar, aar1, linestyle='-', marker='o',  markersize=4, color="Blue")

                plt.grid()
                # plt.xlim([-0.5, 0.5])
                # plt.ylim([-0.5, 1])
                filename = "./image_dir/image_" + str(t) + ".png"
                image_names.append(filename)
                plt.savefig(filename)
                plt.close()
                
            rc_data =  np.random.choice(range(len(self.data)))
            
            bmu = self.fun_getBMU(self.data[rc_data])
            self.listBMU.append(bmu)
            self.fun_updateSOM(bmu,self.data[rc_data],t)
            plt.close()

        images = []
        for n in image_names:
            frame = Image.open(n)
            images.append(frame)

        anim_name = 'animation(h=' + str(self.lattice[0]) + '_w=' + str(self.lattice[1]) + '_LR0=' + str(LR0) + '_stepMax=' + str(stepMax) + ').gif'
        # Save the frames as an animated GIF
        images[0].save(anim_name,
                    save_all=True,
                    append_images=images[1:],
                    duration=800,
                    loop=0)

        animation = pyglet.resource.animation(anim_name)
        sprite = pyglet.sprite.Sprite(animation)

        # create a window and set it to the image size
        win = pyglet.window.Window(width=sprite.width, height=sprite.height)
        green = 0, 0, 0, 1
        pyglet.gl.glClearColor(*green)

        @win.event
        def on_draw():
            win.clear()
            sprite.draw()

        pyglet.app.run()

        # err_x, err_y = np.asarray(error_list).T
        # values = range(len(err_x))
        # plt.plot(values, err_y, linestyle='-', marker='*', markersize=5, color="Blue")
        # plt.xticks(values,err_x)
        # plt.show()
        # plt.close()
        
    def fun_getBMU(self, input_vec):
        listBMU = []
        for row in range(self.lattice[0]):
            for col in range(self.lattice[1]):
                dist = np.linalg.norm((input_vec-self.som[col,row]))
                listBMU.append(((col,row),dist))
        listBMU.sort(key=lambda x: x[1])
        return listBMU[0][0]
                
    def fun_updateSOM(self,bmu,input_vec,t):
        for col in range(self.lattice[0]):
            for row in range(self.lattice[1]):
                distToBMU = np.linalg.norm((np.array(bmu) - np.array((col,row))))
                self.som[(col,row)] += self.fun_neighbouring(distToBMU,t) * self.fun_learning(t) * (input_vec-self.som[(col,row)])

    def fun_learning(self, t):
        lr = self.LR0*np.exp(-t/self.lamLN/2) 
        self.LR.append((t,lr))
        return lr

    def fun_neighbouring(self,distToBMU,t):
        curr_sigma = self.NR0*np.exp(-t/self.lamLN*2) 
        nr = np.exp(-(distToBMU**2)/(2*curr_sigma**2))
        self.NR.append((t,curr_sigma))
        return nr

    def fun_quant_err(self):
        print("Calculating quant error")
        bmuDist = []
        for iter in self.data:
            bmu = self.fun_getBMU(iter)
            bmuFeat = self.som[bmu]
            bmuDist.append(np.linalg.norm(iter-bmuFeat))
 
        err = np.array(bmuDist).mean()
        print("quantization error:", err)
        return err

    def fun_maxdist(self):     
        Rows = self.lattice[0]
        Cols = self.lattice[1]
        maxDist = 0
        for row in range(Rows):
            for col in range(Cols):
                if row-1 >= 0:   
                    dist = np.linalg.norm(self.som[row][col] - self.som[row-1][col]); 
                if row+1 <= Rows-1:   
                    dist = np.linalg.norm(self.som[row][col] - self.som[row+1][col]); 
                if col-1 >= 0:   
                    dist = np.linalg.norm(self.som[row][col] - self.som[row][col-1]); 
                if col+1 <= Cols-1:   
                    dist = np.linalg.norm(self.som[row][col] - self.som[row][col+1]); 
                if dist>maxDist:
                    maxDist = dist
        return maxDist + 1e-3

    def u_matrix(self):
        print("Constructing U-Matrix from SOM")
        Rows = self.lattice[0]
        Cols = self.lattice[1]
       
        umatrix = np.zeros(shape=(Rows, Cols), dtype=np.float64)
        for row in range(Rows):
            for col in range(Cols):
                sumDists = 0.0; cnt = 0
                if row-1 >= 0:   
                    sumDists += np.linalg.norm(self.som[row][col] - self.som[row-1][col]); cnt += 1
                if row+1 <= Rows-1:   
                    sumDists += np.linalg.norm(self.som[row][col] - self.som[row+1][col]); cnt += 1
                if col-1 >= 0:   
                    sumDists += np.linalg.norm(self.som[row][col] - self.som[row][col-1]); cnt += 1
                if col+1 <= Cols-1:   
                    sumDists += np.linalg.norm(self.som[row][col] - self.som[row][col+1]); cnt += 1

                umatrix[row][col] = sumDists / cnt
                
        self.umatrix = umatrix
        print("U-Matrix constructed \n")    
        plt.figure(figsize=(8, 8))
        plt.imshow(umatrix, cmap='gray')
        plt.show() 
        return 

    def plotInputData(self):
        plt.figure(figsize=(8, 8))
        plt.plot(self.data[:,0], self.data[:,1], linestyle='', marker='*',  markersize=5, color="Red")
        plt.grid()
        # plt.xlim([-0.5, 0.5])
        # plt.ylim([-0.5, 1])
        plt.show()

    def plotSOM(self):
        arr = []    
        arr1 = []
        for iter in self.som:
            arr = np.hstack((arr, iter[:,0]))
            arr1 = np.hstack((arr1, iter[:,1]))

        xx, yy = np.meshgrid(arr, arr1)
        plt.figure(figsize=(8, 8))
        plt.plot(arr, arr1, linestyle='', marker='o',  markersize=2, color="Blue")
        plt.grid()
        # plt.xlim([-0.5, 0.5])
        # plt.ylim([-0.5, 1])
        plt.show()         

    def plotDensity(self, data, linewidth = 4):
        Rows = self.lattice[0]
        Cols = self.lattice[1]
        densitylist = np.zeros((Rows, Cols), dtype=int)
        for bmu in self.listBMU:
            densitylist[bmu[0],bmu[1]] += 1

        maxDist = self.fun_maxdist()

        plt.figure(figsize=(8, 8)) 
        color = 0
        for row in range(Rows):
            for col in range(Cols):
                if row-1 >= 0:    # above
                    dist = np.linalg.norm(self.som[row][col] - self.som[row-1][col])
                    color = dist/maxDist + 0.1    
                    if color > 1 :
                        color = 1
                    plt.plot([row-0.5, row-0.5], [col-0.5, col+0.5], color=str(color), linewidth=linewidth) 
                if row+1 <= Rows-1:   # below
                    dist = np.linalg.norm(self.som[row][col] - self.som[row+1][col])
                    color = dist/maxDist    
                    if color > 1 :
                        color = 1
                    plt.plot([row+0.5, row+0.5], [col-0.5, col+0.5], color=str(color), linewidth=linewidth)
                if col-1 >= 0:   # left
                    dist = np.linalg.norm(self.som[row][col] - self.som[row][col-1])
                    color = dist/maxDist    
                    if color > 1 :
                        color = 1
                    plt.plot([row-0.5, row+0.5], [col-0.5, col-0.5], color=str(color), linewidth=linewidth)
                if col+1 <= Cols-1:   # right
                    dist = np.linalg.norm(self.som[row][col] - self.som[row][col+1])
                    color = dist/maxDist    
                    if color > 1 :
                        color = 1
                    plt.plot([row-0.5, row+0.5], [col+0.5, col+0.5], color=str(color), linewidth=linewidth)
                if dist>maxDist:
                    maxDist = dist     

        colors = [(0.7, 0, 0), (0, 0, 0)]  # R -> B
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'my_cmap'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

        plt.imshow(densitylist, cmap=cmap)
        plt.colorbar()
        plt.show()
        return

    def plotRateDecay(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        n_NR = np.array(self.NR)
        l_NR = np.array(self.LR)
        ax1.plot(l_NR[:,0], l_NR[:,1])
        ax2.plot(n_NR[:,0], n_NR[:,1])
        ax1.title.set_text('Learning Rate Decay')
        ax2.title.set_text('Neighborhood Rate Decay')
        fig.tight_layout()
        plt.show()
        plt.close()

def PointsInCircum(r,n=5000):
    return np.array([[math.cos(2*np.pi/n*x)*r+0.5,math.sin(2*np.pi/n*x)*r+0.5] for x in range(0,n+1)])

def main():
    # circle_data = PointsInCircum(0.5)

    # read RBF Data from xlsx file
    RBFdata = pd.read_excel('RBF_Data.xlsx')
    RBFdata = RBFdata.drop(['Label'], axis=1) 
    square = RBFdata.divide(RBFdata.max().max())
    square_data = square.to_numpy()

    som_square = SOM(30,30,2,square_data)
    som_square.fun_trainSOM(LR0=0.7,lamLN=300,NR0=10,stepMax=1000)

    # # calculate and print U-Matrix
    # som_square.u_matrix()
    # som_square.plotInputData()
    # som_square.plotSOM()
    som_square.plotDensity(square_data, linewidth=4)
    # som_square.plotRateDecay()

if __name__=="__main__":
  main()
