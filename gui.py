# coding:utf-8
from typing import re

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import qtawesome
from PyQt5.QtGui import QBrush, QPixmap, QPalette, QTextCursor
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
from glob import glob
import imageio

# class EmittingStr(QtCore.QObject):
#     textWritten = QtCore.pyqtSignal(str) #定义一个发送str的信号
#     def write(self, text):
#       self.textWritten.emit(str(text))
#       loop = QEventLoop()
#       QTimer.singleShot(1000, loop.quit)
#       loop.exec_()
from PyQt5.QtWidgets import QLabel, QApplication, QTextBrowser

class SortImg():
    def tryint(self, s):
        try:
            return int(s)
        except ValueError:
            return s

    def str2int(self, v_str):
        return [self.tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

    def sort_humanly(self, v_list):
        return sorted(v_list, key=self.str2int)

class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
    #     sys.stdout = EmittingStr(textWritten=self.outputWritten)
    #     sys.stderr = EmittingStr(textWritten=self.outputWritten)
    #
    # def outputWritten(self, text):
    #     cursor = self.process.textCursor()
    #     cursor.movePosition(QtGui.QTextCursor.End)
    #     cursor.insertText(text)
    #     self.process.setTextCursor(cursor)
    #     self.process.ensureCursorVisible()

    def printf(self, mypstr):
        """
        自定义类print函数, 借用c语言
        printf
        Mypstr：是待显示的字符串
        """

        self.process.append(mypstr)  # 在指定的区域显示提示信息
        self.cursor = self.process.textCursor()
        self.process.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QtWidgets.QApplication.processEvents()  # 一定加上这个功能，不然有卡顿


    """
    QWidget *widget // 待添加的子窗口
    int fromRow     // 横坐标
    int fromColumn  // 纵坐标
    int rowSpan     // 横向跨越几个单元格
    int columnSpan  // 纵向跨越几个单元格
    """

    def button3clicked(self):
        self.esrganwithcycle()

    def init_ui(self):
        self.setFixedSize(1290, 800)
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap('./litah.png')))
        self.setPalette(palette)

        self.left_up_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_up_widget.setObjectName('left_up_widget')
        self.left_up_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_up_widget.setLayout(self.left_up_layout)  # 设置左侧部件布局为网格

        self.left_down_widget = QtWidgets.QWidget()
        self.left_down_widget.setObjectName('left_down_widget')
        self.left_down_layout = QtWidgets.QGridLayout()
        self.left_down_widget.setLayout(self.left_down_layout)

        self.right_widget = QtWidgets.QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格

        self.main_layout.addWidget(self.left_up_widget, 0, 0, 3, 3)  # 左侧部件在第0行第0列，占12行2列
        self.main_layout.addWidget(self.left_down_widget, 4, 0, 8, 3)
        self.main_layout.addWidget(self.right_widget, 0, 3, 12, 9)  # 右侧部件在第0行第3列，占12行9列


        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        self.left_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.music', color='red'), "srgan")
        self.left_button_1.setObjectName('left_button')
        self.left_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.sellsy', color='blue'), "esrgan")
        self.left_button_2.setObjectName('left_button')
        self.left_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.film', color='yellow'), "esrgan with cyclegan")
        self.left_button_3.setObjectName('left_button')


        self.process = QtWidgets.QTextBrowser(self)
        self.process.ensureCursorVisible()
        self.process.setLineWrapColumnOrWidth(500)
        # self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)
        self.process.setFixedWidth(600)
        self.process.setFixedHeight(800)
        self.process.move(200, 50)
        self.process.append("poverty freedom")

        # self.left_layout.addWidget(self.left_label_1, 1, 0, 1, 3)
        self.left_up_layout.addWidget(self.left_button_1, 0, 0, 1, 3)
        self.left_up_layout.addWidget(self.left_button_2, 1, 0, 1, 3)
        self.left_up_layout.addWidget(self.left_button_3, 2, 0, 1, 3)
        # self.left_layout.addWidget(self.left_label_2, 5, 0, 1, 3)
        # self.left_layout.addWidget(self.left_button_4, 6, 0, 1, 3)
        # self.left_layout.addWidget(self.left_button_5, 7, 0, 1, 3)
        # self.left_layout.addWidget(self.left_button_6, 8, 0, 1, 3)
        # self.left_layout.addWidget(self.left_label_3, 9, 0, 1, 3)
        # self.left_layout.addWidget(self.left_button_7, 10, 0, 1, 3)
        # self.left_layout.addWidget(self.left_button_8, 11, 0, 1, 3)
        # self.left_layout.addWidget(self.left_button_9, 12, 0, 1, 3)

        self.pixmap = QPixmap('./pic/welcome.png')

        self.left_down_text = QtWidgets.QTextBrowser(self)
        self.left_down_text.ensureCursorVisible()
        self.left_down_text.setFixedWidth(617)
        self.left_down_text.setFixedHeight(50)
        self.left_down_text.append(
            "一共使用三种模型作对比实验,得到的结果分别用pixel loss,content loss,PSNR,SSIM,MSSIM加以评判")

        self.right_layout.addWidget(self.process)

        self.left_button_1.clicked.connect(lambda : self.button1clicked())
        self.left_button_2.clicked.connect(lambda : self.button2clicked())
        self.left_button_3.clicked.connect(lambda : self.button3clicked())

        self.lab = QLabel()
        self.lab.setFixedHeight(300)
        self.lab.setFixedWidth(650)
        self.lab.setPixmap(self.pixmap)
        self.left_down_layout.addWidget(self.lab)
        self.left_down_layout.addWidget(self.left_down_text)

        self.process.setStyleSheet(
            '''QLineEdit{
                    border:1px solid gray;
                    width:300px;
                    border-radius:10px;
                    padding:2px 4px;
            }''')
        self.main_widget.setStyleSheet('''
                            QWidget#widget{
                                color:#232C51;
                                background:white;
                                border-top:1px solid darkGray;
                                border-bottom:1px solid darkGray;
                                border-right:1px solid darkGray;
                            }
                            QLabel#lable{
                                border:none;
                                font-size:16px;
                                font-weight:800;
                                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                            }
                            QPushButton#button:hover{border-left:4px solid red;font-weight:700;}
                        ''')
    def button1clicked(self):
        pixmap = QPixmap('pic/srgan.PNG')
        lab = QLabel()
        lab.setPixmap(pixmap)
        left_down_text = QtWidgets.QTextBrowser(self)
        left_down_text.ensureCursorVisible()
        left_down_text.setFixedWidth(625)
        left_down_text.setFixedHeight(100)
        left_down_text.append(
            "1/将SRResNet作为生成网络的GAN模型用于超分，即SRGAN模型。这是首篇在人类感知视觉上进行超分的文章，而以往的文章以PSNR为导向，但那些方式并不能让人眼觉得感知到了高分辨率——Photo-Realistic")
        print("learn well")
        QApplication.processEvents()
        print("leran better")
        item_list = list(range(self.left_down_layout.count()))
        item_list.reverse()  # 倒序删除，避免影响布局顺序

        for i in item_list:
            item = self.left_down_layout.itemAt(i)
            self.left_down_layout.removeItem(item)
            if item.widget():
                item.widget().deleteLater()

        # self.left_down_layout.removeItem(self.pixmap)
        # self.left_down_layout.removeItem(self.lab)
        self.left_down_layout.addWidget(lab)
        self.left_down_layout.addWidget(left_down_text)
        self.lab = lab
        self.left_down_text = left_down_text
        print("asdasd")
        app.processEvents()
    def button2clicked(self):
        pixmap = QPixmap('./pic/esrgan.png')
        lab = QLabel()
        lab.setPixmap(pixmap)
        left_down_text = QtWidgets.QTextBrowser(self)
        left_down_text.ensureCursorVisible()
        left_down_text.setFixedWidth(625)
        left_down_text.setFixedHeight(100)
        left_down_text.append(
            "相较于SRGAN,主要提出了SRResNet网络结构的改进：移除BN，有利于去除伪影，提升泛化能力;使用Residual-in-Residual Dense Block (RRDB)作为基本构建模块，更强更易训练;GAN-based Network的损失函数的改进：使用RaGAN (Relativistic average GAN)中的相对损失函数，提升图像的相对真实性从而恢复更多的纹理细节；感知损失函数的改进：使用VGG激活层前的特征值计算重构损失，提升了亮度的一致性和纹理恢复程度。")

        QApplication.processEvents()
        item_list = list(range(self.left_down_layout.count()))
        item_list.reverse()  # 倒序删除，避免影响布局顺序

        for i in item_list:
            item = self.left_down_layout.itemAt(i)
            self.left_down_layout.removeItem(item)
            if item.widget():
                item.widget().deleteLater()

        # self.left_down_layout.removeItem(self.pixmap)
        # self.left_down_layout.removeItem(self.lab)
        self.left_down_layout.addWidget(lab)
        self.left_down_layout.addWidget(left_down_text)
        self.lab = lab
        self.left_down_text = left_down_text
        app.processEvents()

    def button3clicked(self):
        pixmap = QPixmap('./pic/esrganwithcycle.png')
        lab = QLabel()
        lab.setPixmap(pixmap)
        left_down_text = QtWidgets.QTextBrowser(self)
        left_down_text.ensureCursorVisible()
        left_down_text.setFixedWidth(617)
        left_down_text.setFixedHeight(50)
        left_down_text.append(
            "在该模型中一共有两个 GAN，它们的结构相同，功能不同，一个是从 LR 图像生成 HR 图像，然后做对抗学习；另一个是从 HR 图像生成 LR 图像，然后做对抗学习。")
        QApplication.processEvents()
        item_list = list(range(self.left_down_layout.count()))
        item_list.reverse()  # 倒序删除，避免影响布局顺序

        for i in item_list:
            item = self.left_down_layout.itemAt(i)
            self.left_down_layout.removeItem(item)
            if item.widget():
                item.widget().deleteLater()

        self.left_down_layout.addWidget(lab)
        self.left_down_layout.addWidget(left_down_text)
        self.lab = lab
        self.left_down_text = left_down_text
        app.processEvents()

    def readFile(self):
        results = []
        with open('./log/srgan/loss.txt','r') as file:
            line = file.readline()
            line = file.readline()
            temp = []
            while line:
                tmp = line.split(' ')
                temp.append(tmp)
                line = file.readline(' ')
            results.append(temp)

        with open('./log/esrgan/loss.txt','r') as file:
            line = file.readline()
            line = file.readline()
            temp = []
            while line:
                tmp = line.split(' ')
                temp.append(tmp)
                line = file.readline(' ')
            results.append(temp)

        with open('./log/esrganwithcycle/loss.txt','r') as file:
            line = file.readline()
            line = file.readline()
            temp = []
            while line:
                tmp = line.split(' ')
                temp.append(tmp)
                line = file.readline(' ')
            results.append(temp)
        return results


    def arrange(self):
        self.right_lab = []
        self.right_lab_button = []
        self.lb = {'SRGAN','ESRGAN','ESRGANWITHCYCLE'}
        self.right_lab_name = [['pixel','content',''],['','',''],['','','']]
        self.right_lab_value = []
        self.results = self.readFile()

        with open('./log/srgan/loss.txt') as file:
            self.len = file.readlines()

        for it in range(self.len):
            for i in range(3):
                tmplab = QLabel()
                tmplab.setText(self.lb[i])
                self.right_layout.addWidget(tmplab,i*4,0,1,9)
                tmp = []
                start = i*4+1
                for j in range(3):
                    for k in range(3):
                        tmpname = QLabel()
                        tmpname.setText(self.right_lab_name[j][k])
                        tmpvalue = QTextBrowser()
                        tmpvalue.append(self.results[i][j*3+k])
                        self.right_layout.addWidget(tmpname,i*4+j+1,k*3,1,1)
                        self.right_layout.addWidget(tmpvalue,i*4+j+1,k*3+1,1,2)
        app.processEvents()

    def drawLoss(self):
        writer = SummaryWriter('./board', comment='selection of a bunch of pictures')
        y = []
        x = []
        for i in range(1,self.results[0].size()+1,1):
            x.append(i)
            y.append([])
        for i in range(self.results[0][0].size()):
            y[0].append([])
            y[1].append([])
            y[2].append([])

        for i in range(3):
            for j in range(self.results[0].size()):
                for k in range(self.results[0][0].size()):
                    y[i][k].append(self.results[i][j][k])

        for i in range(3):
            plt.figure( figsize=(1080, 1080), facecolor='red', edgecolor='black')
            plt.title('matplotlib.pyplot.figure() Example\n',
                      fontsize=14, fontweight='bold')
            plt.rcParams['font.family'] = 'MicroSoft YaHei'  # 设置字体，默认字体显示不了中文
            for j in range(self.results[0].size()):
                plt.plot(x,y[i][j],label=self.right_lab_name[j])
            plt.savefig('./loss_pictures/' + "%s" % self.lb[i])



            # for i in x:
            #     writer.add_scalars('trainning loss',{'high-res pixel-wise loss against ground truth':pixel_h[i],
            #                                          'low-res pixel-wise loss against ground truth':pixel_l[i],
            #                                          'high-res cycle loss':cycle_h[i],
            #                                          'low-res cycle loss': cycle_l[i],
            #                                          'generator loss':g[i],
            #                                          'f generator loss':f[i]
            #                                         },i)
            #     writer.add_histogram('pixel_h',pixel_h[i],i)
            #     writer.add_histogram('pixel_l',pixel_l[i],i)
            #     writer.add_histogram('cycle_h',cycle_h[i],i)
            #     writer.add_histogram('cycle_l',cycle_l[i],i)
            #     writer.add_histogram('g',g[i],i)
            #     writer.add_histogram('f',f[i],i)
            # for i in xt:
            #     writer.add_scalars('test loss',{'high-res pixel-wise loss':pixel_ht[i],
            #                                     'low-res pixel-wise loss':pixel_lt[i],
            #                                     'high-res discriminator loss':d_h[i],
            #                                     'low-res discriminator loss':d_l[i]
            #                                     })
            #     writer.add_histogram('pixel_ht',pixel_ht[i],i)
            #     writer.add_histogram('pixel_lt',pixel_lt[i],i)
            #     writer.add_histogram('d_h',d_h[i],i)
            #     writer.add_histogram('d_l',d_l[i],i)
            #
            # writer.close()):
        """
        with open('./log/'+dir+'/loss.txt', 'a') as file:
            file.write(loss_pixel_h.item() + ' ')
            file.write(loss_content_h.item() + ' ')
            file.write(loss_G_h.item()+' ')
            file.write(loss_G.item()+' ')
            file.write(loss_D.item()+' ')
            file.write(mse+' ')
            file.write(10 * np.log10(1.0 * 255 * 255 / loss_psnr) + ' ')
            file.write(ssimxy+' ')
            file.write(mssim)    
        """

        for ii in range(3):
            writer = SummaryWriter('./board/train'+self.lb[ii], comment='selection of a bunch of pictures')
            for i in self.results[ii][0].size():
                writer.add_scalars(self.lb[ii],{     'pixel-wise loss against ground truth':self.results[ii][0][i],
                                                     'content loss against ground truth':self.results[ii][1][i],
                                                     'adversial loss':self.results[ii][2][i],
                                                     'generator loss': self.results[ii][3][i],
                                                     'discriminator loss':self.results[ii][4][i],
                                                     'MSE loss':self.results[ii][5][i],
                                                     'PSNR loss':self.results[ii][6][i],
                                                     'SSIM loss':self.results[ii][7][i],
                                                     'Mean-SSIM loss':self.results[ii][8][i]
                                                },i)
                writer.add_histogram('pixel loss',self.results[ii][0][i],i)
                writer.add_histogram('content loss',self.results[ii][1][i],i)
                writer.add_histogram('adversial loss',self.results[ii][2][i],i)
                writer.add_histogram('generator loss',self.results[ii][3][i],i)
                writer.add_histogram('discriminator loss',self.results[ii][4][i],i)
                writer.add_histogram('mse loss',self.results[ii][5][i],i)
                writer.add_histogram('psnr loss',self.results[ii][6][i],i)
                writer.add_histogram('ssim loss',self.results[ii][7][i],i)
                writer.add_histogram('meanssim loss',self.results[ii][8][i],i)
                # writer.add_histogram('pixel_h',pixel_h[i],i)
                # writer.add_histogram('pixel_l',pixel_l[i],i)
                # writer.add_histogram('cycle_h',cycle_h[i],i)
                # writer.add_histogram('cycle_l',cycle_l[i],i)
                # writer.add_histogram('g',g[i],i)
                # writer.add_histogram('f',f[i],i)
            # for i in xt:
            #     writer.add_scalars('test loss',{'high-res pixel-wise loss':pixel_ht[i],
            #                                     'low-res pixel-wise loss':pixel_lt[i],
            #                                     'high-res discriminator loss':d_h[i],
            #                                     'low-res discriminator loss':d_l[i]
            #                                     })
            #     writer.add_histogram('pixel_ht',pixel_ht[i],i)
            #     writer.add_histogram('pixel_lt',pixel_lt[i],i)
            #     writer.add_histogram('d_h',d_h[i],i)
            #     writer.add_histogram('d_l',d_l[i],i)

            writer.close()

    def makegif(self):
        def readssim(file):
            ans = []
            with open(file,'r') as f:
                line = f.readline()
                while line:
                    ans.append(line)
            return ans

        def drawssim(filepath, dir ,value,epoch):
            # 打开图像
            im = Image.open(filepath)

            # 告诉系统，你要在图像上画画了
            draw = ImageDraw.Draw(im)

            # 设置字体，大小=50
            font = ImageFont.truetype('C:\\Windows\\Fonts\\arial.ttf', 40)
            # 如何找系统上自带的字体： 百度搜索Linux字体存放位置

            # 写内容, 初始位置(30, 10) 颜色蓝色
            draw.text((30, 10), f'ssim = {value}', fill='blue', font=font)
            draw.text((30,20) , f'epoch= {epoch}', fill='red' , font=font)
            # 保存
            im.save(filepath.replace('pic', 'pic_final'))

        self.filesrgan = sorted(glob.glob("./validation/srgan"+"/*.*"))
        self.fileesrgan = sorted(glob.glob("./validation/esrgan"+"/*.*"))
        self.fileesrganwithcycle = sorted(glob.glob("./validation/esrganwithcycle"+"/*.*"))

        self.ssim1 = readssim('./validation/srgan/text')
        self.ssim2 = readssim('./validation/esrgan/text')
        self.ssim3 = readssim('./validation/esrganwithcycle/text')

        count = 0
        for img in self.filesrgan:
            drawssim(img,'srgan',self.ssim1[count],count)
            drawssim(img,'esrgan',self.ssim2[count],count)
            drawssim(img,'esrganwithcycle',self.ssim3[count],count)
            count += 1

        frames1 = []
        frames2 = []
        frames3 = []

        self.imgs1 = glob('./validation/srgan/pic_final'+'/*.*')
        self.imgs2 = glob('./validation/esrgan/pic_final'+'/*.*')
        self.imgs3 = glob('./validation/esrganwithcycle/pic_final'+'/*.*')

        self.sorted_imgs1 = SortImg().sort_humanly(self.imgs1)
        self.sorted_imgs2 = SortImg().sort_humanly(self.imgs2)
        self.sorted_imgs3 = SortImg().sort_humanly(self.imgs3)

        for img in self.sorted_imgs1:
            frames1.append(imageio.imread(img))
        for img in self.sorted_imgs2:
            frames2.append(imageio.imread(img))
        for img in self.sorted_imgs3:
            frames3.append(imageio.imread(img))

        imageio.mimsave('srgan', frames1, 'GIF', duration=1.5)
        imageio.mimsave('esrgan', frames2, 'GIF', duration=1.5)
        imageio.mimsave('esrganwithcycle', frames3, 'GIF', duration=1.5)

# ssh -p 54909 root@region-3.autodl.com
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())