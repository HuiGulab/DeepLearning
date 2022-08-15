# DeepLearning

深度学习的学习笔记库

# 准备

Github账户(将用户名发给我)

下载GitHub Desktop或Git GUI对仓库进行代码管理

# 参与人员

目前内容包括研一同学推送学习文档笔记和实践测试

# 要求

此仓库不定期更新，为确保推送拉取操作不产生冲突，在上传学习笔记前先执行```git pull``` 获取***当前分支***在远程仓库更新后的状态

使用git指令拉取本仓库后，切换到属于自己的分支，根据[实际使用情况进行对应的操作](#git控制仓库的工具和部分指令)。在每周三把学习章节按照文件夹结构的方式创建或更新当前正在学习笔记，然后推送到远程仓库中。[相关说明文档](https://docs.github.com/cn)

常规操作指令

	添加修改后的文件 git add .									
	批注此次修改更新的信息 git commit -a "   "							
	将此次更新推送到远程仓库 git push									



## 学习书籍

[书籍链接](https://github.com/XiangLinPro/IT_book)

解压密码x20200427


## 学习任务

任务文档格式

推荐学习使用[markdown格式](https://www.markdownguide.org/basic-syntax)编辑长文本，也是常见的开源程序中readme文档的书写格式

任务内容

该仓库下的isses功能区域不定时更新学习任务信息，并且指定参与成员进行任务。以深度学习电子书的章节为单位，可通过Jupyter Notebook、其他形式的```python.py``` 执行文件记录学习笔记并保存，**每周三**推送到远程仓库中




## 文件夹结构

括号内的中文字符仅做说明示例，可省略

例

	├── git
	├── GuoxuSheng
		├── DiveIntoDeepLearning(深度学习电子书)
			├── Chapter2
				├── 2预备知识.ipynb
			├── Chapter3
				├── 3线性神经网络.ipynb
				······
			├── Chapter15
				├── 15自然语言处理.ipynb
		├── DL&ML(天池实践测试)


## Git控制仓库的工具和部分指令
### GitHub Desktop

可通过点击```current branch```的下拉菜单切换分支

### Git Bash

查看所有分支```git branch -a```并根据显示的remote/origin/姓名，新建本地分支并同步远程分支。例如```git checkout -b GuoxuSheng origin/GuoxuSheng```即可

如从main分支切换到其他分支时遇到```error: Your local changes to the following files would be overwritten by checkout:```，依次执行

	git add .
	git commit -a "commit message"

将main分支修改的文件暂存到缓存区，即可正常切换分支


---------------------------------------------
