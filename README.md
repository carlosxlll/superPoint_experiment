# superPoint_experiment

### git 操作流程
```bash
# 操作实例:假如我们在dev分支上开发完项目，执行完下面的代码
git add .
git commit -m '备注信息'
gut push -u origin dev

# 想把dev分支合并到master分支上
# 1.切换master分支 2.pull远程master代码 3.dev合并到master上
git checkout master
git pull origin master
git merge dev
# 4.查看状态 5，提交master新信息
git status
git push origin master
```








#### 说明手册
```bash
# 查看分支(本地|远程|所有)
git branch
git branch -r
git branch -a

# 创建分支
git branch master2
git checkout -b master2

# 切换分支
git checkout master

# 对比分支
git diff master...master2

# 合并分支(将master2合并到当前分支)
git merge master2
```