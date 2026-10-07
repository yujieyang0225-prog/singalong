FROM innatera.azurecr.io/sdk:4.7.1

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
# USER root

ARG REMOTE_USER
ARG REMOTE_UID=9000
ARG REMOTE_GID=$REMOTE_UID

RUN addgroup --gid ${REMOTE_GID} ${REMOTE_USER} \
    && adduser --disabled-password --uid ${REMOTE_UID} --gid ${REMOTE_GID} ${REMOTE_USER} \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $REMOTE_USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$REMOTE_USER \
    && chmod 0440 /etc/sudoers.d/$REMOTE_USER

# dependencies of Innatera's wristband package (gesture_recognition_imu, mounted from the starter repo),
# which we import for GRSnn2LayersRNN and the quantizers; versions from its pyproject.toml
RUN pip install --no-cache-dir seaborn "scikit-learn==1.6.0" "torchsummary==1.5.1" "pyzenodo3==1.0.2" \
    "pyelftools==0.31" pretty_midi

ENV HOME=/home/${REMOTE_USER}
ENV PATH=${HOME}/.local/bin:${PATH}
USER ${REMOTE_USER}
