# DeepLearning

深度学习的学习笔记库

# 准备

Github账户(将用户名发给我)

Git GUI或GitHub Desktop

# 要求

以学习章节为单位，通过Jupyter Notebook记录学习笔记，**每周三**推送到远程仓库中

## 文件夹结构

例

	├── GuoxuSheng
		├── Chapter2
			├── 2预备知识.ipynb
			├── 3线性神经网络.ipynb
			······
			├── 15自然语言处理.ipynb

## 执行操作

使用git指令拉取本仓库后，查看所有分支```git branch -a```并根据显示的姓名，切换到属于自己的分支，例如```git checkout GuoxuSheng```。在每周三时把章节按照文件夹结构的方式创建或更新当前正在学习笔记，然后推送到远程仓库中

## Git Bash

如从main分支切换到其他分支时遇到```error: Your local changes to the following files would be overwritten by checkout:```，依次执行

	git add .
	git commit -a "commit message"

将main分支修改的文件暂存到缓存区，即可正常切换分支

## GitHub Desktop

可通过点击```current branch```的下拉菜单切换分支

---------------------------------------------
