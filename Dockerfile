# If the training machine is not GPU-enabled, use:
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-0
# Otherwise, use:
#FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-3
WORKDIR /root

RUN pip install pandas numpy google-cloud-storage scikit-learn opencv-python
RUN apt-get update; apt-get install git -y; apt-get install -y libgl1-mesa-dev

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git clone https://github.com/sergiovirahonda/AutomaticTraining-Dataset.git

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git clone https://github.com/sergiovirahonda/AutomaticTraining-DataCommit.git

RUN mv /root/AutomaticTraining-DataCommit/task.py /task.py
RUN mv /root/AutomaticTraining-DataCommit/data_utils.py /data_utils.py
RUN mv /root/AutomaticTraining-DataCommit/email_notifications.py /email_notifications.py

ENTRYPOINT ["python","task.py"]
