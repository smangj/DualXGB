# LongXGB

## 环境配置

### 初始化及依赖包管理

#### 运行环境初次配置步骤：
***如果已在当前项目下创建过虚拟环境，请先手动删除虚拟环境目录，例如`venv`目录***：
1. 在PyCharm终端中执行 ```pdm install```，如果正确配置了系统全局python的话，pdm会自动利用系统全局python为当前项目创建虚拟环境目录`.venv`，同时安装好所有依赖包。
2. 在PyCharm里配置解释器路径为：
   - Windows: `{项目根目录}/.venv/Scripts/python.exe`
   - Mac/Linux: `{项目根目录}/.venv/bin/python`
3. 在PyCharm终端中执行 ```pre-commit install```初始化pre-commit模块

#### 依赖包列表更新流程：
1. 对于程序运行所需的依赖包，使用```pdm add {包名} {包名} ... --save-exact```命令添加，以pandas和flask为例：
    ```
    pdm add pandas flask --save-exact
    ```
2. 对于开发过程中所需，但程序运行非必须的包可使用```pdm add -dG dev {包名} {包名} ... --save-exact```命令添加至开发环境依赖列表。例如`jupyter`、`black`等包都属于开发工具，算法程序在服务器上运行并不依赖它们。
    ```
    pdm add -dG dev jupyter black --save-exact
    ```
3. 依赖包添加完成后，`pdm`会更新`pdm.lock`和`pyproject.toml`文件，请提交到git中。