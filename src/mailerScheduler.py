from restService import RestService

class MailerScheduler:
    def __init__(self):
        self.receivers = {}
        self.buffer = 10

    def send_mail(self, receiver, time, classification):
        if not self.check_if_send(receiver, time):
            self.receivers[receiver] = time
            body = {
                'email': 'id@studiodvd.co.il',
                'subject': f"Sending mail to {receiver} with classification {classification}",
                'message': f"Send mail to {receiver} with classification {classification}"
            }
            RestService('send_email').post(body)

        else:
            print(f"Mail to {receiver} is already sent")

    def check_if_send(self, receiver, time):
        if receiver in self.receivers:
            time_to_send = self.receivers[receiver]
            if time_to_send < time + self.buffer:
                return True
            else:
                return False
        return False
