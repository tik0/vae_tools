#!/usr/bin/python
from IPython.core.display import display, HTML

def notebook_resize(width = 70, margin_left = 1):
    display(HTML("<style>.container { width:" + str(int(width)) + "% !important; margin-left:" + str(int(margin_left))+ "% !important; }</style>"))


