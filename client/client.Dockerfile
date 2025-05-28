# FROM node:18-alpine
FROM node:14

# 安裝 node-sass 編譯需要的工具
RUN apt-get update && apt-get install -y python3 make g++

WORKDIR /usr/src/frontend

COPY package*.json ./

RUN npm install
RUN npm install --save \
    @fortawesome/fontawesome-svg-core \
    @fortawesome/free-solid-svg-icons \
    @fortawesome/vue-fontawesome

COPY . .

EXPOSE 8080

CMD ["npm", "run", "serve"]