FROM node:24-bookworm-slim AS build
WORKDIR /app/10_app_react
COPY 10_app_react/package.json 10_app_react/package-lock.json ./
RUN npm ci
COPY 10_app_react/ ./
COPY 03_src/ReactTabs/ /app/03_src/ReactTabs/
RUN npm run build

FROM node:24-bookworm-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production PORT=5173
COPY --from=build /app/10_app_react/package.json /app/10_app_react/package-lock.json ./
RUN npm ci --omit=dev
COPY --from=build /app/10_app_react/dist ./dist
COPY --from=build /app/10_app_react/server.mjs ./
USER node
EXPOSE 5173
CMD ["node", "server.mjs"]
