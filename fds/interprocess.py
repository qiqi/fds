from multiprocessing import Manager

lock = Manager.lock()
env = manager.dict()
