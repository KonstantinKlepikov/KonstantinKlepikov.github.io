from collections import namedtuple

FakeUser = namedtuple('user', 
                      ['id', 'email'], 
                      defaults=['omnomnom0@example.com']
                      )

class MyNewFakeUser(FakeUser):

    def remove_email(self):
        if self.email == 'omnomnom0@example.com':
            return 'omnomnom'
            
this = MyNewFakeUser(0)

print(this)
this.remove_email()