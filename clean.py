import os
import getopt

#sudo python3 clean.py 清理 

def clean_path(path):
    print("before clean:", '==='*20)
    os.system("sudo du -sh " + path)
    os.system("sudo rm -rf " + path)
    print("after clean:")
    os.system("sudo du -sh " + path)


def clean():
    paths = ["/System/Library/Speech/Voices/", "/private/var/log/",
             "/private/var/folders/", "/private/var/tmp/TM*", "~/Library/Caches/"]
    for path in paths:
        clean_path(path)


def disk_total():
    print('loading ...')
    os.system("sudo du -sh /*")
    print("===="*20)
    os.system("sudo df -h")


def print_menu():
    print("""
        *************************
        a : 自定义目录清洗
        c : 开始自动清洗
        d : 查询磁盘空间
        q : 退出
        *************************
    """)


def menu():
    print_menu()
    char = input("请选择: ")
    while char != "q":
        if char == 'a':
            path = input("请输入路径:")
            clean_path(path)
        elif char == 'c':
            clean()
        elif char == 'd':
            disk_total()
        else:
            print("没有此选项!")
        print_menu()
        char = input("请选择: ")
    print('By!')


if __name__ == "__main__":
    menu()
