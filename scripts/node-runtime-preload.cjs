// Some restricted Windows Node installations cannot resolve the local account
// through os.userInfo(). tsx only needs the username to name its temp folder.
const os = require('node:os');
try {
  os.userInfo();
} catch {
  os.userInfo = () => ({ username: process.env.USERNAME || 'local' });
}
