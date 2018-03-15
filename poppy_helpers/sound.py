import paramiko

class RemoteSound(object):
    def __init__(self, host, user="poppy", passw="poppy"):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=host, username=user, password=passw)

    def play(self, sound):
        stdin, stdout, stderr = self.ssh.exec_command("aplay ~/{}.wav".format(sound))

        return stdout.read()

    def __del__(self):
        self.ssh.close()

