# -*- coding: UTF8 -*-

import os
import sys
from time import time

from utils import Search
from PyQt5.QtWidgets import *

page_header = ['新闻标题', '附件数', '权限要求']
file_header = ['文件名', '所属新闻标题', '权限要求']
authority = ['角色 1', '角色 2', '角色 3', '角色 4']


class Retrieval(QWidget):
    def __init__(self, table, search, mode):
        super().__init__()
        self.mode = mode
        self.setWindowTitle('实验三-企业检索系统')
        self.search_button = QPushButton('查询')
        # 查询输入框
        self.input_box = QLineEdit(self)
        # 查询条数和时间显示
        self.search_result_label = QLabel(self)
        # 角色选择
        self.combo = QComboBox()
        # 查询结果表
        self.table = table
        self._setup()
        # 查询结果
        self.lines = []
        self.search = search
        self.show()

    def _setup(self):
        # set layout
        search_layout = QHBoxLayout()
        search_layout.addWidget(self.combo)
        search_layout.addWidget(self.input_box)
        search_layout.addWidget(self.search_button)
        window_layout = QVBoxLayout()
        window_layout.addLayout(search_layout)
        window_layout.addWidget(self.search_result_label)
        window_layout.addWidget(self.table)
        self.setLayout(window_layout)
        # 设定查询按钮
        self.search_button.clicked.connect(self.__search)
        # 设定角色
        self.combo.addItems(authority)
        # 设定双击打开文章
        self.table.itemDoubleClicked.connect(self.__open)

    def get_role(self):
        role = self.combo.currentText()
        return int(role[-1])

    def __search(self):
        query = self.input_box.text()
        self.table.clear()
        if self.mode == 'page':
            self.table.setHorizontalHeaderLabels(page_header)
        else:
            self.table.setHorizontalHeaderLabels(file_header)
        # 按照新闻标题进行自动扩展
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        if self.mode == 'file':
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        # 只能单选
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        # 只能选取行
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # 禁止点击表头
        self.table.horizontalHeader().setSectionsClickable(False)
        # 禁止修改表格内容
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 查询计时
        start_time = time()
        self.lines = self.search.search(query, self.get_role(), self.mode)
        end_time = time()
        self.table.setRowCount(len(self.lines))
        self.search_result_label.setText('找到 {:d} 条结果, 耗时 {:.6f}s'.format(len(self.lines), end_time - start_time))
        if self.mode == 'page':
            for row, data in enumerate(self.lines):
                self.table.setItem(row, 0, QTableWidgetItem(data['title']))
                self.table.setItem(row, 1, QTableWidgetItem(str(len(data['file_name']))))
                self.table.setItem(row, 2, QTableWidgetItem('至少角色{:d}'.format(data['authority'])))
        else:
            for row, file in enumerate(self.lines):
                self.table.setItem(row, 0, QTableWidgetItem(file[0]))
                self.table.setItem(row, 1, QTableWidgetItem(file[1]))
                self.table.setItem(row, 2, QTableWidgetItem('至少角色{:d}'.format(file[2])))

    def __open(self, item):
        if self.mode == 'page':
            pos = item.row()
            print("double clicked " + str(pos))
            data = self.lines[pos]
            QMessageBox.about(self, data['title'], 'title : {:s}\n\nurl : {:s}\n\nparagraphs : {:s}\n\n'
                                                   'file_name : {:s}\n\n'
                                                   'authority : 角色{:d}\n\n'
                              .format(data['title'], data['url'], data['paragraphs'],
                                      str(data['file_name']), data['authority']))
        else:
            pos = item.row()
            print("double clicked " + str(pos))
            data = self.lines[pos]
            path:str = data[0]
            # 使用 utf-8 操控 Windows cmd
            os.system('chcp 65001')
            os.system('start '+path)


class RetrievalSystem(QTabWidget):

    def __init__(self):
        super(RetrievalSystem, self).__init__()
        self.setWindowTitle('实验三-企业检索系统')

        self.resize(1280, 720)

        table1 = QTableWidget(self)
        table1.setColumnCount(3)
        table1.setHorizontalHeaderLabels(page_header)

        table2 = QTableWidget(self)
        table2.setColumnCount(3)
        table2.setHorizontalHeaderLabels(file_header)

        search = Search()
        self.tab1 = Retrieval(table1, search, 'page')
        self.tab2 = Retrieval(table2, search, 'file')
        self.addTab(self.tab1, "页面检索")
        self.addTab(self.tab2, "文档检索")
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RetrievalSystem()
    sys.exit(app.exec_())
