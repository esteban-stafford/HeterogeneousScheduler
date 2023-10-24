FROM python:3.7.17

WORKDIR /usr/src/app
COPY requirements.txt ./

RUN apt-get update &&  \
    apt-get install -y libopenmpi-dev
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e git+https://github.com/openai/gym.git@cedecb35e3428985fd4efad738befeb75b9077f1#egg=gym
RUN pip install -e git+https://github.com/openai/spinningup.git@2e0eff9bd019c317af908b72c056a33f14626602#egg=spinup

COPY . .

# CMD [ "python", "ppo-pick-jobs.py" ]
CMD [ "./run_experiment.sh" ]
