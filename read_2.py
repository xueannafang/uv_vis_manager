# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:26:04 2020

@author: Xue Fang
"""
import os

import xlsxwriter


path = os.getcwd()

def get_dir(name):
    dir_path = '\\'.join([path, name])
    if os.path.exists(dir_path):
        return dir_path
    else:
        raise ValueError(f'No file or directory has name {name}')

def get_target(dir_path, prefix=''):
    dir_list = [f for f in os.listdir(dir_path) 
                if os.path.isdir('\\'.join([dir_path, f]))
                and f.startswith(prefix)]
    files = {}
    for d in dir_list:
        target_path = '\\'.join([dir_path, d])
        file_list = [f for f in os.listdir(target_path) 
                     if os.path.isfile('\\'.join([target_path, f]))
                     and f.endswith('.txt')]
        files[d] = file_list
    return files

def parse(filepath):
    with open(filepath, 'r') as f:
        text = f.read().splitlines()[1:]
    first_line = text.pop(0).split('\t')
    content = [line.split('\t') for line in text]
    return first_line, content

def write_xlsx(name, prefix=''):
    dir_path = get_dir(name)
    files = get_target(dir_path, prefix=prefix)
    workbook = xlsxwriter.Workbook(name+'.xlsx')
    for sheet_name in files.keys():
        worksheet = workbook.add_worksheet(sheet_name)
        file_list = files[sheet_name]
        col = 0
        for filename in file_list:
            filepath = '//'.join([dir_path, sheet_name, filename])
            first_line, content = parse(filepath)
            worksheet.write(0, col, filename)
            worksheet.write(1, col, first_line[0])
            worksheet.write(1, col+1, first_line[1])
            for i, element in enumerate(content):
                row = i+2
                worksheet.write(row, col, element[0])
                worksheet.write(row, col+1, element[1])
            col += 2
    workbook.close()
            
            