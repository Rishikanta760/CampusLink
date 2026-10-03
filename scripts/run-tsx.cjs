const path = require('node:path');
const { spawnSync } = require('node:child_process');

const preload = path.resolve(__dirname, 'node-runtime-preload.cjs');
const cli = path.resolve(__dirname, '../node_modules/tsx/dist/cli.mjs');
const env = { ...process.env };
const preloadOption = `--require ${JSON.stringify(preload.replace(/\\/g, '/'))}`;
env.NODE_OPTIONS = [env.NODE_OPTIONS, preloadOption].filter(Boolean).join(' ');
const result = spawnSync(process.execPath, [cli, ...process.argv.slice(2)], { stdio: 'inherit', env });
if (result.error) throw result.error;
process.exit(result.status ?? 1);
