import os
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

data_header = ['选择', '主题', 'url', '附件数', '权限要求']
file_header = ['选择', '文件名', '所属主题', '权限要求']
authority = ['角色 1', '角色 2', '角色 3', '角色 4']

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ATTACHMENT_PATH = os.path.join(DIR_PATH, 'data', 'attachment')


class Retrieval(QWidget):
    def __init__(self, table, retriever):
        super().__init__()
        self.setWindowTitle('实验三-企业检索系统')
        self.search_button = QPushButton('查询')
        self.input_box = QLineEdit(self)  # 查询输入框
        self.search_result_label = QLabel(self)
        self.combo = QComboBox()
        self.table = table

        self._setup()

        self.lines = []
        self.check_box = []
        self.retriever = retriever
        self.show()

    def _setup(self):
        # set layout
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.combo)
        hbox1.addWidget(self.input_box)
        hbox1.addWidget(self.search_button)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.table)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addWidget(self.search_result_label)
        vbox.addLayout(hbox2)
        self.setLayout(vbox)
        # set listener
        self.search_button.clicked.connect(self._search)
        # self.open_button.clicked.connect(self._open)
        # set combo
        self.combo.addItems(authority)

    def _get_role(self):
        role = self.combo.currentText()
        return int(role[-1])

    def _search(self):
        pass

    def _open(self):
        pass


class DataTab(Retrieval):
    def __init__(self, table, retriever):
        super().__init__(table, retriever)

    def _search(self):
        query = self.input_box.text()
        self.check_box = []
        self.table.clear()
        self.table.setHorizontalHeaderLabels(data_header)
        self.lines = self.retriever.search_data(query, self._get_role())
        self.table.setRowCount(len(self.lines))
        self.search_result_label.setText('找到 %d 条结果' % len(self.lines))

        for row, data in enumerate(self.lines):
            ck = QCheckBox()
            self.check_box.append(ck)
            hbox = QHBoxLayout()
            hbox.setAlignment(Qt.AlignCenter)
            hbox.addWidget(ck)
            w = QWidget()
            w.setLayout(hbox)

            # '选择', '主题', 'url', '附件数', '权限要求'
            self.table.setCellWidget(row, 0, w)
            self.table.setItem(row, 1, QTableWidgetItem(data['title']))
            self.table.setItem(row, 2, QTableWidgetItem(data['url']))
            self.table.setItem(row, 3, QTableWidgetItem(str(len(data['file_name']))))
            self.table.setItem(row, 4, QTableWidgetItem('>= %d' % data['level']))

    def _open(self):
        choosed_data = [self.lines[i] for i, ck in enumerate(self.check_box) if ck.isChecked()]
        for data in choosed_data:
            QMessageBox.about(self, data['title'], 'title : %s\n\n'
                                                   'url : %s\n\n'
                                                   'parapraghs : %s\n\n'
                                                   'file_name : %s\n\n'
                                                   'privilege level : %d\n\n'
                              % (data['title'], data['url'], data['paragraphs'], str(data['file_name']), data['level']))

        for ck in self.check_box:
            ck.setChecked(False)


class FileTab(Retrieval):
    def __init__(self, table, retriever):
        super().__init__(table, retriever)

    def _search(self):
        query = self.input_box.text()
        self.check_box = []
        self.table.clear()
        self.table.setHorizontalHeaderLabels(file_header)
        self.lines = self.retriever.search_file(query, self._get_role())
        self.table.setRowCount(len(self.lines))
        self.search_result_label.setText('找到 %d 条结果' % len(self.lines))

        for row, file in enumerate(self.lines):
            ck = QCheckBox()
            self.check_box.append(ck)
            hbox = QHBoxLayout()
            hbox.setAlignment(Qt.AlignCenter)
            hbox.addWidget(ck)
            w = QWidget()
            w.setLayout(hbox)

            # '选择', '文件名', '所属主题', '权限要求'
            self.table.setCellWidget(row, 0, w)
            self.table.setItem(row, 1, QTableWidgetItem(file[0]))
            self.table.setItem(row, 2, QTableWidgetItem(file[1]))
            self.table.setItem(row, 3, QTableWidgetItem('>= %d' % file[2]))

    def _open(self):
        choosed_data = [self.lines[i][:2] for i, ck in enumerate(self.check_box) if ck.isChecked()]
        for file, title in choosed_data:
            path = os.path.join(ATTACHMENT_PATH, title, file + '.jpg')
            if not os.path.exists(path):
                QMessageBox.warning(self, '找不到文件', '不存在路径 ' + path, QMessageBox.Ok)
            else:
                pass
                # Image.open(path).show()

        for ck in self.check_box:
            ck.setChecked(False)


class RetrievalSystem(QTabWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
    #     self.initUI()
    #
    # def initUI(self):
        self.setWindowTitle('企业检索系统')

        self.resize(600, 800)

        table1 = QTableWidget(self)
        table1.setColumnCount(5)
        table1.setHorizontalHeaderLabels(data_header)

        table2 = QTableWidget(self)
        table2.setColumnCount(4)
        table2.setHorizontalHeaderLabels(file_header)

        # retriever = Retriever()

        self.tab1 = DataTab(table1, None)
        self.tab2 = FileTab(table2, None)
        self.addTab(self.tab1, "数据检索")
        self.addTab(self.tab2, "文档检索")
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RetrievalSystem()
    sys.exit(app.exec_())

# class RetrievalSystem(QTableWidget):
#
#     def __init__(self):
#         super(RetrievalSystem, self).__init__()
#         self.initUI()
#
#     def initUI(self):
#         self.setWindowTitle("实验三-企业检索系统")
#         self.resize(800, 600)
#         web_tab = QTableWidget(self)
#         web_tab.setColumnCount(5)
#         web_tab.setHorizontalHeaderLabels(data_header)
#
#         file_tab = QTableWidget(self)
#         file_tab.setColumnCount(4)
#         file_tab.setHorizontalHeaderLabels(file_header)
#         self.addTab()
#         self.show()
#
#
# if __name__ == '__main__':
#     window = QApplication(sys.argv)
#     w = RetrievalSystem()
#     sys.exit(window.exec_())
