import smtplib

# Email variables definition
# -----------------------------------------------------------------------------------------------
sender = 'svirahonda@gmail.com' #replace this by your email address
receiver = ['svirahonda@gmail.com'] #replace this by the owner's email address
smtp_provider = 'smtp.gmail.com'
smtp_port = 587
smtp_account = 'svirahonda@gmail.com' #replace this by your stmp account email address
smtp_password = '' #replace this by your smtp account password
# -----------------------------------------------------------------------------------------------

def training_result(result,accuracy):

    if result == 'old_evaluation_prod':
        message = "A data commit has been detected, therefore old model from /production has been evaluated and has reached more than 0.90 of accuracy. There's no need to retrain it, Check the GCP logs for more information."
    if result == 'retrain_prod':
        message = 'A data commit has been detected, therefore old model from /production has been retrained and after evaluation has reached more than 0.90 of accuracy. It has been saved into /testing, Check the GCP logs for more information.'
    if result == 'old_evaluation_test':
        message = "A data commit has been detected, therefore old model from /testing has been evaluated and has reached more than 0.90 of accuracy. There's no need to retrain it, Check the GCP logs for more information."
    if result == 'retrain_test':
        message = 'A data commit has been detected, therefore old model from /testing has been retrained and after evaluation has reached more than 0.90 of accuracy. It has been saved into /testing, Check the GCP logs for more information.'
    if result == 'poor_metrics':
        message = 'A data commit has been detected, therefore old models from /production and /testing have been retrained but none of them reached more than 0.90 of accuracy during evauation. Check the GCP logs for more information.'
    if result == 'not_found':
        message = 'No previous models were found at GCS. YOu must Start a training from scratch. Check GCP logs for more information'
    message = 'Subject: {}\n\n{}'.format('An automatic training job has ended recently', message)

    try:
        server = smtplib.SMTP(smtp_provider,smtp_port)
        server.starttls()
        server.login(smtp_account,smtp_password)
        server.sendmail(sender, receiver, message)         
        print('Email sent successfully',flush=True)
        return
    except Exception as e:
        print('Something went wrong. Unable to send email.',flush=True)
        print('Exception: ',e)
        return

def exception(e_message):

    try:
        message = 'Subject: {}\n\n{}'.format('An automatic training job has failed recently', e_message)
        server = smtplib.SMTP(smtp_provider,smtp_port)
        server.starttls()
        server.login(smtp_account,smtp_password)
        server.sendmail(sender, receiver, message)         
        print('Email sent successfully',flush=True)
        return
    except Exception as e:
        print('Something went wrong. Unable to send email.',flush=True)
        print('Exception: ',e)
        return
