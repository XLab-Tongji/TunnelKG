from selenium import webdriver
from selenium.common.exceptions import NoAlertPresentException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time


def print_hi():
    chromeOptions = Options()
    prefs = {'profile.default_content_settings.popups': 0,  # 防止保存弹窗
             'download.default_directory': "E:\\crawler",  # 设置默认下载路径
             "profile.default_content_setting_values.automatic_downloads": 1  # 允许多文件下载
             }

    chromeOptions.add_experimental_option("prefs", prefs)
    browser = webdriver.Chrome(options=chromeOptions)
    url = "http://www.xdsdjs.com/CN/volumn/volumn_1164.shtml"
    browser.get(url)

    # Use a breakpoint in the code line below to debug your script.

    browser.maximize_window()

    # ifr = browser.find_element_by_xpath("//iframe[@id='UserInfo']")
    # browser.switch_to.frame(ifr)

    # lo = browser.find_element_by_xpath("//input[@name='login_id']")
    # ps = browser.find_element_by_xpath("//input[@name='remote_password']")
    # lo.send_keys('skybluecat')
    # ps.send_keys('zxcvbnm')
    # browser.find_element_by_xpath("//input[@name='Submit2']").click()

    # time.sleep(2)
    # browser.find_element_by_xpath("//a[@class='current']").click()
    # time.sleep(2)

    # inputs = browser.find_element_by_xpath("//input[@name='ks_keyword']")  # 找到搜索框
    # inputs.send_keys('隧道')  # 传送入关键词
    # browser.find_element_by_xpath("//input[@name='Submit22']").click()
    # time.sleep(5)
    # browser.switch_to.window(browser.window_handles[-1])

    # s=browser.find_elements_by_xpath("//a[@class='J_WenZhang']")[1].click()
    # s = browser.find_elements_by_xpath("//a[@class='J_WenZhang']")[2].click()
    # print(len(s))
    # browser.find_elements_by_xpath("//a[@class='J_WenZhang']")[1].click()
    # for z in range(0, 38):
    #     browser.find_elements_by_xpath("//a[@class='J_WenZhang']")[2].click()
    now = 1164
    for i in range(0, 0):
        pre = "//a[@href='volumn_" + str(now - 1) + ".shtml']"
        print(pre)
        browser.find_element_by_xpath(pre).click()
        time.sleep(1)
        now = now - 1

    while 1:
        allE = browser.find_elements_by_xpath("//a[@class='J_VM']")
        print(len(allE))
        for j in range(0, 90):
            if j % 3 == 2:
                allE[j].click()
                time.sleep(2)
                try:
                    a = browser.switch_to.alert
                    a.accept()
                except NoAlertPresentException:
                    pass
            time.sleep(3)
        time.sleep(1)
        pre = "//a[@href='volumn_" + str(now - 1) + ".shtml']"
        now = now - 1
        browser.find_element_by_xpath(pre).click()
        time.sleep(1)

    # for i in range(0, 49):
    #    allE = browser.find_elements_by_xpath("//a[@class='J_WenZhang_U' and @href='#']")
    #    for j in range(0, len(allE)):
    #        allE[j].click()
    #        time.sleep(2)
    #        try:
    #            a = browser.switch_to.alert
    #            a.accept()
    #        except NoAlertPresentException:
    #            pass
    #        time.sleep(5)
    #    time.sleep(8)
    #    if i == 0:
    #        browser.find_elements_by_xpath("//a[@class='J_WenZhang']")[1].click()
    #        time.sleep(1)
    #    else:
    #        browser.find_elements_by_xpath("//a[@class='J_WenZhang']")[2].click()
    #    time.sleep(1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()