#coding=utf-8

import MySQLdb
import time
import json

configfile = './config.json'
jsondata = json.load(file(configfile))
dbip = jsondata['dbip']
dbport = jsondata['dbport']
dbroot = jsondata['dbroot']
dbname = jsondata['dbname']
dbcharset = jsondata['dbcharset']

class Database(object):
    def __init__(self):
        self.conn = self.initDatabase()

    def initDatabase(self):
        '''
        初始化数据库
        TODO 数据库失败
        '''
        conn = MySQLdb.connect(
            host=dbip,
            port = dbport,
            user = dbroot,
            passwd = '!@#asd123',
            db = dbname,
            charset=dbcharset,
        )
        return conn


    def qureyTaskname(self, ownername):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        cur = self.conn.cursor()
        if ownername=='all':
            sql = "select name, createtime, process, status, type,  usrname from Tasks where status = 1 and exclusworker = 0"
        else:
            sql = "select name, createtime, process, status, type,  usrname from Tasks where status = 1 and usrname = '%s' and exclusworker = 1"%ownername
        #sql = "describe tags"
        #print sql
        # 执行sql语句
        cur.execute(sql)
        res = cur.fetchall()
        #print 'result:', res
        #cur.close()
        #self.conn.close()
        out = []
        for r in res:
            out.append(r)
        return out

    def sqlwrite(self, cur, name, t, process, usrname):
        
        sql = "insert Tasks(name, createtime, process, usrname) \
                values('%s', '%s', '%lf', '%s')" % \
                (name, t, process, usrname)
        #print sql
        try:
            # 执行sql语句
            cur.execute(sql)
            # 提交到数据库执行
            self.conn.commit()
        except:
            #print 'error'
            #print sql
            # 发生错误时回滚
            self.conn.rollback()
        return 

    def writeTask(self, name, process, usrname):
        '''
        将结果写入数据库
        '''
        ISOTIMEFORMAT='%Y-%m-%d %X'
        t = time.strftime( ISOTIMEFORMAT, time.localtime() )
        cur = self.conn.cursor()
        self.sqlwrite(cur, name, t, process, usrname)
        cur.close()
        return

    def delTask(self, usrname, taskname):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        cur = self.conn.cursor()
        sql = "delete from Tasks where usrname = '%s' and name = '%s'"%(usrname, taskname)
        #sql = "describe tags"
        #print sql
        # 执行sql语句
        cur.execute(sql)
        self.conn.commit()
        return

    def startTask(self, usrname, taskname):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        # todo task status ==2???
        cur = self.conn.cursor()
        sql = "update  Tasks set status = 1, stop = 0  where usrname = '%s' and name = '%s'"%(usrname, taskname)
        #sql = "describe tags"
        #print sql
        # 执行sql语句
        cur.execute(sql)
        self.conn.commit()
        return

    def updateTask(self, usrname, taskname, percent, pid):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        cur = self.conn.cursor()
        sql = "update Tasks set process = %lf, workerpid = %d where  name = '%s' and usrname = '%s'"%(percent, pid, taskname, usrname)
        #sql = "describe tags"
        # print sql
        # 执行sql语句
        cur.execute(sql)
        self.conn.commit()
        return 

    def getTaskpid(self, usrname, taskname):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        cur = self.conn.cursor()
        sql = "select  workerpid from Tasks where  name = '%s' and usrname = '%s'"%(taskname, usrname)
        #sql = "describe tags"
        # print sql
        # 执行sql语句
        cur.execute(sql)
        res = cur.fetchall()

        return res

    def getTaskStatus(self, usrname, taskname):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        cur = self.conn.cursor()
        sql = "select  status, stop from Tasks where  name = '%s' and usrname = '%s'"%(taskname, usrname)
        #sql = "describe tags"
        # print sql
        # 执行sql语句
        cur.execute(sql)
        res = cur.fetchall()

        return res

    def startTaskTrain(self, usrname, taskname):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        # todo task status ==2???
        cur = self.conn.cursor()
        sql = "update  Tasks set status = 2, trained = 1  where usrname = '%s' and name = '%s'"%(usrname, taskname)
        #sql = "describe tags"
        print sql
        # 执行sql语句
        cur.execute(sql)
        self.conn.commit()
        return

    def FinishTaskTrain(self, usrname, taskname):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        # todo task status ==2???
        cur = self.conn.cursor()
        sql = "update  Tasks set status = 3, stop = 0, workerpid = 0 where usrname = '%s' and name = '%s'"%(usrname, taskname)
        #sql = "describe tags"
        #print sql
        # 执行sql语句
        cur.execute(sql)
        self.conn.commit()
        return

    def TaskTrainError(self, usrname, taskname):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        # todo task status ==2???
        cur = self.conn.cursor()
        sql = "update  Tasks set status = 4  where usrname = '%s' and name = '%s'"%(usrname, taskname)
        #sql = "describe tags"
        #print sql
        # 执行sql语句
        cur.execute(sql)
        self.conn.commit()
        return

    def stopTask(self, usrname, taskname):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        # todo task status ==2???
        cur = self.conn.cursor()
        sql = "update  Tasks set status = 0, process = 0.0, workerpid = 0, stop = 1 where usrname = '%s' and name = '%s'"%(usrname, taskname)
        #sql = "describe tags"
        #print sql
        # 执行sql语句
        cur.execute(sql)
        self.conn.commit()
        return

    def clearTask(self, usrname, taskname):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        # todo task status ==2???
        cur = self.conn.cursor()
        sql = "update  Tasks set status = 0, process = 0.0, workerpid = 0, stop = 0 where usrname = '%s' and name = '%s'"%(usrname, taskname)
        #sql = "describe tags"
        #print sql
        # 执行sql语句
        cur.execute(sql)
        self.conn.commit()
        return

    def addWorker(self, name, gpustatus, memunused):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        ISOTIMEFORMAT='%Y-%m-%d %X'
        t = time.strftime( ISOTIMEFORMAT, time.localtime() )
        cur = self.conn.cursor()
        sql = "select * from Workers where name= '%s'"%(name)
        cur.execute(sql)
        res = cur.fetchall()
        if len(res)==0:
            cur = self.conn.cursor()
            sql = "insert into Workers(name, status, taskname, GPUMemery, Memusage, updatetime, usrname)values('%s', 0, 'None', '%s', %d, '%s', 'all')"%(name, gpustatus, memunused, t)
            #sql = "describe tags"
            print sql
            # 执行sql语句
            cur.execute(sql)
            self.conn.commit()
        return 

    def insertWorker(self, name, status, taskname, gpustatus, memunused):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        ISOTIMEFORMAT='%Y-%m-%d %X'
        t = time.strftime( ISOTIMEFORMAT, time.localtime() )
        cur = self.conn.cursor()
        sql = "select * from Workers where name= '%s'"%(name)
        cur.execute(sql)
        res = cur.fetchall()
        if len(res)==0:
            cur = self.conn.cursor()
            sql = "insert into Workers(name, status, taskname, GPUMemery, Memusage, updatetime, usrname)values('%s', %d, '%s', '%s', %d, '%s', 'all')"%(name, status, taskname, gpustatus, memunused, t)
            #sql = "describe tags"
            #print sql
            # 执行sql语句
            cur.execute(sql)
            self.conn.commit()
        return 

    def clearWorker(self, taskname):
        ISOTIMEFORMAT='%Y-%m-%d %X'
        t = time.strftime( ISOTIMEFORMAT, time.localtime() )
        cur = self.conn.cursor()
        sql = "update Workers set status = 0,  updatetime = '%s', taskname = 'None' where taskname = '%s'"%(t, taskname)
        #sql = "describe tags"
        print sql
        # 执行sql语句
        cur.execute(sql)
        self.conn.commit()
        return

    def updateWorker(self, name, status, taskname, gpustatus, memunused):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        ISOTIMEFORMAT='%Y-%m-%d %X'
        t = time.strftime( ISOTIMEFORMAT, time.localtime() )
        cur = self.conn.cursor()
        sql = "update Workers set status = %d,  taskname = '%s',  GPUMemery = '%s', Memusage = %d, updatetime = '%s' where name = '%s'"%(status, taskname, gpustatus, memunused, t, name)
        #sql = "describe tags"
        # print sql
        # 执行sql语句
        cur.execute(sql)
        self.conn.commit()
        return 

    def qureyWorkerlist(self, usrname):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        cur = self.conn.cursor()
        sql = "select name, status, taskname, GPUMemery, updatetime from Workers"
        #sql = "describe tags"
        #print sql
        # 执行sql语句
        cur.execute(sql)
        res = cur.fetchall()
        #print 'result:', res
        #cur.close()
        #self.conn.close()
        return res

    def qureyWorkerOwner(self, workername):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        cur = self.conn.cursor()
        sql = "select usrname from Workers where name = '%s'"%(workername)
        #sql = "describe tags"
        #print sql
        # 执行sql语句
        cur.execute(sql)
        res = cur.fetchall()
        #print 'result:', res
        #cur.close()
        #self.conn.close()
        return res

    def qureyWorkerStatus(self):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        cur = self.conn.cursor()
        sql = "select name, status, Memusage, usrname from Workers"
        #sql = "describe tags"
        #print sql
        # 执行sql语句
        cur.execute(sql)
        res = cur.fetchall()
        #print 'result:', res
        #cur.close()
        #self.conn.close()
        return res

    def delWorker(self, name):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        cur = self.conn.cursor()
        sql = "delete from Workers where name = '%s'"%(name)
        #sql = "describe tags"
        #print sql
        # 执行sql语句
        cur.execute(sql)
        self.conn.commit()
        return

    def addusr(self, usrname, email, passwd, active, level, groups):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        cur = self.conn.cursor()
        sql = "select * from Users where name= '%s' or email = '%s'"%(usrname, email)
        cur.execute(sql)
        res = cur.fetchall()
        if len(res)==0:
            sql = "insert into Users(name, email, passwd, active, level, groups)values('%s', '%s', '%s', %d, %d, '%s')"%(usrname, email, passwd, active, level, groups)
            cur.execute(sql)
            self.conn.commit()
            cur.close()
            self.conn.close()
            return True
        cur.close()
        self.conn.close()
        return False

    def verifyusr(self, usrname, passwd):
        """根据查找id
        
        Arguments:
            taskname {[string]} -- [标签名称]
        """ 
        cur = self.conn.cursor()
        sql = "select * from Users where name= '%s' and passwd = '%s'"%(usrname, passwd)
        cur.execute(sql)
        res = cur.fetchall()
        cur.close()
        self.conn.close()
        if len(res)==0:
            return False
        return True

    def operationrecord(self, info, types):
        ISOTIMEFORMAT='%Y-%m-%d %X'
        t = time.strftime( ISOTIMEFORMAT, time.localtime() )
        cur = self.conn.cursor()
        sql = "insert into Operations(operation, types, updatetime)values('%s', '%s', '%s')"%(info, types, t)
        cur.execute(sql)
        self.conn.commit()
        cur.close()
        self.conn.close()

if __name__ == "__main__":
    db = Database()
    #db.qureyTask()
    #db.writeTask('task1', 0.0, 'fj')
    res = db.qureyWorkerlist('fj1')
    print res[0]