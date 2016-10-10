import sys
import numpy as np
import numpy.matlib
from PySide import QtGui, QtCore
from matplotlib.backends import qt_compat
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import integrate
from time import time

class MyTableModel(QtCore.QAbstractTableModel):
	dataChanged = QtCore.Signal()

	def __init__(self, parent, mylist, header, grid, *args):
		QtCore.QAbstractTableModel.__init__(self, parent, *args)
		self.mylist = mylist
		self.header = header
		self.grid = grid

	def rowCount(self, parent):
		if len(self.mylist) > 0:
			return len(self.mylist[0])
		else:
			return 0

	def columnCount(self, parent):
		return len(self.mylist)

	def data(self, index, role):
		if not index.isValid():
			return None
		elif role != QtCore.Qt.DisplayRole:
			return None
		return self.mylist[index.column()][index.row()]

	def headerData(self, col, orientation, role):
		if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole and col < len(self.header):
			return self.header[col]
		return None

	def flags(self, index):
		return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled

	def data(self, index, role):
		if index.isValid():
			if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
				return str(self.mylist[index.column()][index.row()]/self.grid.sizeScalefactor())
		return None

	def setData(self, index, value, role):
		if index.isValid():
			if role == QtCore.Qt.EditRole:
				self.mylist[index.column()][index.row()] = (float)(value)*self.grid.sizeScalefactor()
				self.dataChanged.emit()
				return True
		return False

	def updateView(self):
		self.layoutChanged.emit()

class MyTableModel2(QtCore.QAbstractTableModel):
	dataChanged = QtCore.Signal()

	def __init__(self, parent, mylist, header, *args):
		QtCore.QAbstractTableModel.__init__(self, parent, *args)
		self.mylist = mylist
		self.header = header

	def rowCount(self, parent):
		return len(self.mylist)

	def columnCount(self, parent):
		if len(self.mylist) > 0:
			return len(self.mylist[0])
		else:
			return 0

	def data(self, index, role):
		if not index.isValid():
			return None
		elif role != QtCore.Qt.DisplayRole:
			return None
		return self.mylist[index.row()][index.column()]

	def headerData(self, col, orientation, role):
		if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole and col < len(self.header):
			return self.header[col]
		return None

	def flags(self, index):
		return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled

	def data(self, index, role):
		if index.isValid():
			if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
				return str(self.mylist[index.row()][index.column()])
		return None

	def setData(self, index, value, role):
		if index.isValid():
			if role == QtCore.Qt.EditRole:
				self.mylist[index.row()][index.column()] = (float)(value)
				self.dataChanged.emit()
				return True
		return False

	def updateView(self):
		self.layoutChanged.emit()

class Phase():
	def __init__(self, grid):
		self.grid = grid
		self.table = [[0.00 for _ in range(grid.nx)] for _ in range(grid.ny)]
		self.view = PhaseTab(self)
		self.plot = PhasePlot(self)

class PhaseTab():
	def __init__(self, phase):
		self.phase = phase
		self.prepareTab()

	def replotButtonPressed(self):
		self.phase.plot.updatePlot()

	def prepareTab(self):
		self.tab = QtGui.QWidget()

		layout = QtGui.QGridLayout()

		self.tableView = QtGui.QTableView()
		self.tableModel = MyTableModel2(self.tab, self.phase.table, [])
		self.tableView.setModel(self.tableModel)
		self.tableView.resizeColumnsToContents()

		self.replotButton = QtGui.QPushButton("Replot")
		self.replotButton.clicked.connect(self.replotButtonPressed)

		# self.dxTableView.setMaximumSize(140, 1E5)

		spacer = QtGui.QSpacerItem(10, 50, hPolicy=QtGui.QSizePolicy.Minimum, vPolicy=QtGui.QSizePolicy.Expanding)
		# layout.setColumnStretch(0, 1)

		layout.addWidget(self.replotButton, 0, 0)
		layout.addWidget(self.tableView, 1, 0)
		layout.addItem(spacer, 2, 0)

		self.tab.setLayout(layout)

class PhasePlot():
	def __init__(self, phase):
		self.phase = phase
		self.figure2D = Figure(dpi=72, facecolor=(1,1,1), edgecolor=(0,0,0))
		self.axes2D = self.figure2D.add_subplot(1, 1, 1)
		self.axes2D.set_aspect('equal', adjustable='box')
		self.widget = FigureCanvas(self.figure2D)

	def updatePlot(self):
		self.axes2D.clear()
		self.axes2D.imshow(np.array(self.phase.table), cmap=cm.hot)
		self.figure2D.canvas.draw()

class Amplitude():
	def __init__(self, grid):
		self.grid = grid
		# self.table = [[np.exp(-((i-grid.nx/2.0)/(0.2*grid.nx))**2 - ((j-grid.ny/2.0)/(0.2*grid.ny))**2) for j in range(grid.nx)] for i in range(grid.ny)]
		
		m, n = (5,5)
		self.table = [[np.exp(-((i-grid.nx/(m*2.0))/(0.2*grid.nx/m))**2 - ((j-grid.ny/(n*2.0))/(0.2*grid.ny/n))**2) for j in range(grid.nx/m)] for i in range(grid.ny/n)]
		self.table = np.matlib.repmat(self.table, m, n).tolist()

		# self.table = [[1 for j in range(grid.nx)] for i in range(grid.ny)]
		self.view = AmplitudeTab(self)
		self.plot = AmplitudePlot(self)

class AmplitudeTab():
	def __init__(self, amplitude):
		self.amplitude = amplitude
		self.prepareTab()

	def replotButtonPressed(self):
		self.amplitude.plot.updatePlot()

	def prepareTab(self):
		self.tab = QtGui.QWidget()

		layout = QtGui.QGridLayout()

		self.tableView = QtGui.QTableView()
		self.tableModel = MyTableModel2(self.tab, self.amplitude.table, [])
		self.tableView.setModel(self.tableModel)
		self.tableView.resizeColumnsToContents()
		# self.dxTableView.setMaximumSize(140, 1E5)

		self.replotButton = QtGui.QPushButton("Replot")
		self.replotButton.clicked.connect(self.replotButtonPressed)

		spacer = QtGui.QSpacerItem(10, 50, hPolicy=QtGui.QSizePolicy.Minimum, vPolicy=QtGui.QSizePolicy.Expanding)
		layout.setColumnStretch(0, 1)

		layout.addWidget(self.replotButton, 0, 0)
		layout.addWidget(self.tableView, 1, 0)
		layout.addItem(spacer, 2, 0)

		self.tab.setLayout(layout)

class AmplitudePlot():
	def __init__(self, amplitude):
		self.amplitude = amplitude
		self.figure2D = Figure(dpi=72, facecolor=(1,1,1), edgecolor=(0,0,0))
		self.axes2D = self.figure2D.add_subplot(1, 1, 1)
		# self.axes2D.set_aspect('equal', adjustable='box')
		self.widget = FigureCanvas(self.figure2D)

		im = self.axes2D.imshow(np.array(self.amplitude.table), cmap=cm.hot)
		self.figure2D.colorbar(im)

	def updatePlot(self):
		self.figure2D.clear()
		self.axes2D = self.figure2D.add_subplot(1, 1, 1)
		im = self.axes2D.imshow(np.array(self.amplitude.table).T, cmap=cm.hot)
		self.figure2D.colorbar(im)
		self.figure2D.canvas.draw()

class Grid():
	def __init__(self, sizeX, sizeY, nx, ny):
		self.sizeX = sizeX
		self.sizeY = sizeY
		self.nx = nx
		self.ny = ny
		self.dx = [sizeX/(float)(nx) for _ in range(nx)]
		self.dy = [sizeY/(float)(ny) for _ in range(ny)]
		self.wavelength = 1550E-9
		self.wlUnit = 'nm'
		self.wlScalingDictionary = {'km':1E3, 'm':1, 'mm':1E-3, u"\u00B5m":1E-6, 'nm':1E-9}
		self.view = GridTab(self)

	def sizeScalefactor(self):
		return self.wlScalingDictionary[self.wlUnit]

class GridTab():
	def __init__(self, grid):
		self.grid = grid
		self.prepareTab()

	def addDxRowClicked(self):
		selection = self.dxTableView.selectionModel()
		if selection.hasSelection() and len(selection.selectedIndexes()) > 0:
			pos = selection.selectedIndexes()
			self.grid.dx.insert(pos[0].row()+1, 0)
		else:
			self.grid.dx.append(0)

		self.dxTableModel.updateView()

	def addDyRowClicked(self):
		selection = self.dyTableView.selectionModel()
		if selection.hasSelection() and len(selection.selectedIndexes()) > 0:
			pos = selection.selectedIndexes()
			self.grid.dy.insert(pos[0].row()+1, 0)
		else:
			self.grid.dy.append(0)

		self.dyTableModel.updateView()

	def deleteDxRowClicked(self):
		selection = self.dxTableView.selectionModel()
		if selection.hasSelection() and len(selection.selectedIndexes()) > 0:
			indexes = selection.selectedIndexes()
			rows = []
			for k in range(len(indexes)):
				rows.append(indexes[k].row())	# Can contain duplicates
			uniqueRows = sorted(set(rows), reverse=True)	# Sort reverse, because last row needs to be deleted first
			for row in uniqueRows:
				del self.grid.dx[row]

		self.dxTableModel.updateView()

	def deleteDyRowClicked(self):
		selection = self.dyTableView.selectionModel()
		if selection.hasSelection() and len(selection.selectedIndexes()) > 0:
			indexes = selection.selectedIndexes()
			rows = []
			for k in range(len(indexes)):
				rows.append(indexes[k].row())	# Can contain duplicates
			uniqueRows = sorted(set(rows), reverse=True)	# Sort reverse, because last row needs to be deleted first
			for row in uniqueRows:
				del self.grid.dy[row]

		self.dyTableModel.updateView()

	def wavelengthChanged(self, value):
		self.grid.wavelength = value*self.grid.sizeScalefactor()

	def wavelengthUnitChanged(self, value):
		self.grid.wlUnit = self.wlUnitComboBox.itemText(value)

	def prepareTab(self):
		self.tab = QtGui.QWidget()

		layout = QtGui.QGridLayout()

		self.dxTableView = QtGui.QTableView()
		self.dxTableModel = MyTableModel(self.tab, [self.grid.dx], [u"\u0394x"], self.grid)
		self.dxTableView.setModel(self.dxTableModel)
		self.dxTableView.resizeColumnsToContents()
		self.dxTableView.setMaximumSize(140, 1E5)

		self.dyTableView = QtGui.QTableView()
		self.dyTableModel = MyTableModel(self.tab, [self.grid.dy], [u"\u0394y"], self.grid)
		self.dyTableView.setModel(self.dyTableModel)
		self.dyTableView.resizeColumnsToContents()
		self.dyTableView.setMaximumSize(140, 1E5)

		self.addDxRowButton = QtGui.QPushButton("+")
		self.deleteDxRowButton = QtGui.QPushButton("-")
		self.addDyRowButton = QtGui.QPushButton("+")
		self.deleteDyRowButton = QtGui.QPushButton("-")

		self.addDxRowButton.setMaximumSize(20,20)
		self.deleteDxRowButton.setMaximumSize(20,20)
		self.addDyRowButton.setMaximumSize(20,20)
		self.deleteDyRowButton.setMaximumSize(20,20)

		self.wlUnitLabel = QtGui.QLabel("Wavelength unit ["+self.grid.wlUnit+"]:")
		self.wlLabel = QtGui.QLabel("Wavelength ["+self.grid.wlUnit+"]:")
		self.wlUnitComboBox = QtGui.QComboBox()

		self.wavelength = QtGui.QDoubleSpinBox()
		self.wavelength.setRange(0, 1E8)
		self.wavelength.setDecimals(3)

		wlUnits = ('km', 'm', 'mm', u"\u00B5m", 'nm')
		self.wlUnitComboBox.addItems(wlUnits)
		for k in range(len(wlUnits)):
			if wlUnits[k] == self.grid.wlUnit:
				self.wlUnitComboBox.setCurrentIndex(k)
				break

		self.wavelength.setValue(self.grid.wavelength/self.grid.sizeScalefactor())

		self.wavelength.valueChanged.connect(self.wavelengthChanged)
		self.wlUnitComboBox.currentIndexChanged.connect(self.wavelengthUnitChanged)
		self.addDxRowButton.clicked.connect(self.addDxRowClicked)
		self.deleteDxRowButton.clicked.connect(self.deleteDxRowClicked)
		self.addDyRowButton.clicked.connect(self.addDyRowClicked)
		self.deleteDyRowButton.clicked.connect(self.deleteDyRowClicked)

		spacerV = QtGui.QSpacerItem(10, 50, hPolicy=QtGui.QSizePolicy.Minimum, vPolicy=QtGui.QSizePolicy.Expanding)
		# spacerH = QtGui.QSpacerItem(10, 50, hPolicy=QtGui.QSizePolicy.Minimum, vPolicy=QtGui.QSizePolicy.Expanding)
		layout.setColumnStretch(0, 1)
		layout.addWidget(self.wlUnitLabel, 0, 0, QtCore.Qt.AlignRight)
		layout.addWidget(self.wlUnitComboBox, 0, 1)

		layout.addWidget(self.wlLabel, 1, 0, QtCore.Qt.AlignRight)
		layout.addWidget(self.wavelength, 1, 1)

		layout.addWidget(self.dxTableView, 0, 2, 3, 1)
		layout.addWidget(self.addDxRowButton, 0, 3)
		layout.addWidget(self.deleteDxRowButton, 1, 3)

		layout.addWidget(self.dyTableView, 0, 4, 3, 1)
		layout.addWidget(self.addDyRowButton, 0, 5)
		layout.addWidget(self.deleteDyRowButton, 1, 5)
		# layout.addItem(spacerH, 1, 6)
		layout.addItem(spacerV, 6, 1)

		self.tab.setLayout(layout)

class FF():
	def __init__(self, amplitude, phase, distance, resolution=100):
		self.amplitude = amplitude
		self.phase = phase
		self.distance = distance
		self.resolution = resolution
		self.theta = np.linspace(-np.pi/2, np.pi/2, resolution)
		self.r = np.linspace(0, 2*np.pi, resolution, dtype=complex)
		self.view = FFTab(self)
		self.plot = FFPlot(self)
		self.alpha = 0

	def xToi(self, x):
		for k in range(len(self.distanceX)-1):
			if x > self.distanceX[k] and x < self.distanceX[k+1]:
				return k
		return -1

	def yToj(self, y):
		for k in range(len(self.distanceY)-1):
			if y > self.distanceY[k] and y < self.distanceY[k+1]:
				return k
		return -1

	def h(self, x1, y1, x0, y0, z0):
		d = np.sqrt((x1-x0)**2 + (y1-y0)**2 + z0**2)
		wl = self.distance.wavelength
		return 1j/wl*z0/d*np.exp(-1j*2*np.pi/wl*d)/d

	def U(self, x1, y1):
		i = self.xToi(x1)
		j = self.yToj(y1)
		if i != -1 and j != -1:
			return self.amplitude.table[i][j]*np.exp(1j*self.phase.table[i][j]/180*np.pi) 
		return 0

	def f(self, x1, y1, x0, y0, z0):
		return self.U(x1, y1)*self.h(x1, y1, x0, y0, z0)

	def computeFF(self, alpha):
		self.distanceTP = 1E6*self.distance.wavelength 	# Distance test point
		self.distanceX = np.cumsum(self.distance.dx)
		self.centerX = self.distanceX - np.array(self.distance.dx)/2.0
		self.distanceY = np.cumsum(self.distance.dy)
		self.centerY = self.distanceY - np.array(self.distance.dy)/2.0

		beta = np.linspace(-np.pi/2, np.pi/2, self.resolution)
		x0 = self.distanceTP*np.sin(beta)*np.cos(self.alpha/180*np.pi)+self.distanceX[-1]/2.0
		y0 = self.distanceTP*np.sin(beta)*np.sin(self.alpha/180*np.pi)+self.distanceY[-1]/2.0
		z0 = self.distanceTP*np.cos(beta)

		# for k in range(len(beta)):
		# 	integral = 0
		# 	for i in range(len(self.centerX)):
		# 		for j in range(len(self.centerY)):
		# 			integral += self.f(self.centerX[i], self.centerY[j], x0[k], y0[k], z0[k])*self.distance.dx[i]*self.distance.dy[j]
		# 	self.r[k] = integral

		U = np.array(self.amplitude.table)*np.exp(1j*np.array(self.phase.table)/180*np.pi)
		wl = self.distance.wavelength
		self.centerX = np.matlib.repmat(self.centerX, len(self.distanceY), 1)
		self.centerY = np.matlib.repmat(self.centerY, len(self.distanceX), 1).T

		self.dx = np.matlib.repmat(self.distance.dx, len(self.distanceY), 1)
		self.dy = np.matlib.repmat(self.distance.dy, len(self.distanceX), 1).T
		dA = self.dx * self.dy
		for k in range(len(beta)):
			d = np.sqrt((self.centerX-x0[k])**2 + (self.centerY-y0[k])**2 + z0[k]**2)
			h = 1j/wl*z0[k]/d*np.exp(-1j*2*np.pi/wl*d)/d
			self.r[k] = np.sum(np.sum(U*h*dA))

		self.r = self.r/np.amax(abs(self.r))
		self.plot.updatePlot()

class FFTab():
	def __init__(self, FF):
		self.FF = FF
		self.prepareTab()

	def computeButtonPressed(self):
		self.FF.computeFF(self.alphaValue.value)
		self.FF.plot.updatePlot()

	def alphaChanged(self, value):
		# print value
		self.alphaValue.setValue(value)
		self.FF.alpha = value

	def alphaValueChanged(self, value):
		self.alpha.setValue(value)
		self.FF.alpha = value

	def prepareTab(self):
		self.tab = QtGui.QWidget()

		layout = QtGui.QGridLayout()

		self.alphaValue = QtGui.QDoubleSpinBox()
		self.alpha = QtGui.QSlider(QtCore.Qt.Horizontal)
		self.alphaValue.setRange(0, 180)
		self.alphaValue.setDecimals(3)
		self.alpha.setRange(0, 180)
		self.computeButton = QtGui.QPushButton("compute")

		self.alpha.valueChanged.connect(self.alphaChanged)
		self.alphaValue.valueChanged.connect(self.alphaValueChanged)
		self.computeButton.clicked.connect(self.computeButtonPressed)

		spacer = QtGui.QSpacerItem(10, 50, hPolicy=QtGui.QSizePolicy.Minimum, vPolicy=QtGui.QSizePolicy.Expanding)
		layout.setColumnStretch(0, 1)

		layout.addWidget(self.alphaValue, 0, 0)
		layout.addWidget(self.alpha, 0, 1)
		layout.addWidget(self.computeButton, 1, 0)
		layout.addItem(spacer, 2, 0)

		self.tab.setLayout(layout)

class FFPlot():
	def __init__(self, FF):
		self.FF = FF
		self.figure2D = Figure(dpi=72, facecolor=(1,1,1), edgecolor=(0,0,0))
		self.axes2D = self.figure2D.add_subplot(111, projection='polar')
		# self.axes2D.set_aspect('equal', adjustable='box')
		self.widget = FigureCanvas(self.figure2D)

	def updatePlot(self):
		self.axes2D.clear()
		self.axes2D.plot(self.FF.theta, abs(self.FF.r), color='r', linewidth=3)
		# self.axes2D.imshow(np.array(self.FF.table).T, cmap=cm.hot)
		self.figure2D.canvas.draw()

class NFToFF(QtGui.QWidget):
	def __init__(self):
		super(NFToFF, self).__init__()
		self.initUI()

	def initUI(self):
		layout = QtGui.QGridLayout()
		self.setLayout(layout)

		self.inputTabs = QtGui.QTabWidget()
		self.outputTabs = QtGui.QTabWidget()

		self.grid = Grid(5E-6, 5E-6, 50, 50)
		self.amplitude = Amplitude(self.grid)
		self.phase = Phase(self.grid)
		self.FF = FF(self.amplitude, self.phase, self.grid)

		self.inputTabs.addTab(self.grid.view.tab, "Settings")
		self.inputTabs.addTab(self.amplitude.view.tab, "Amplitude")
		self.inputTabs.addTab(self.phase.view.tab, "Phase")
		self.inputTabs.addTab(self.FF.view.tab, "Far Field")

		self.outputTabs.addTab(self.amplitude.plot.widget, "Amplitude")
		self.outputTabs.addTab(self.phase.plot.widget, "Phase")
		self.outputTabs.addTab(self.FF.plot.widget, "Far Field")

		self.hSplitter = QtGui.QSplitter(QtCore.Qt.Horizontal, self)
		self.hSplitter.addWidget(self.inputTabs)
		self.hSplitter.addWidget(self.outputTabs)
		layout.addWidget(self.hSplitter, 0, 0)

		self.resize(1200, 700)
		self.center()
		self.setWindowTitle('Near to far field solver')		# Later on replaced by the name of the project
		self.show()

	def center(self):
		qr = self.frameGeometry()
		cp = QtGui.QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())	

if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	window = NFToFF()
	sys.exit(app.exec_())