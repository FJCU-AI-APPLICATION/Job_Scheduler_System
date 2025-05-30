# 第一階段：建置 Vue 應用
FROM node:14 as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# 第二階段：用 Nginx 服務 Vue app
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY ./prod.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]