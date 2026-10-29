/* Gridworld RL lecture engine: environment, planners, learners, drawing and one "scene" per animated slide.
   Shared by every slide through layouts/demo.vue. Scenes are looked up by id (see SCENE_IDS at the bottom). */

/* Seeded RNG: each slide reseeds on entry and on reset (r), so live runs replay identically. */
export const SEED=42;let rngState=SEED;
export function reseed(k=0){rngState=(SEED*2654435761+k*40503)>>>0}
function rand(){let t=rngState=(rngState+0x6D2B79F5)>>>0;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}

/* Playback state shared with the layout and keyboard shortcuts */
let playing=!(typeof matchMedia!=='undefined'&&matchMedia('(prefers-reduced-motion: reduce)').matches);
let active=null;
export const transport={get playing(){return playing},set playing(v){playing=v},get speed(){return P.speed},set speed(v){P.speed=Math.max(.25,Math.min(6,v))}};
/* ================= Environment ================= */
const LAYOUT=["....G",".#..L",".#.#.",".....","S.L.."];
const R=5,C=5,N=R*C,S0=4*C+0;
const ARW=['↑','→','↓','←'], DR=[-1,0,1,0], DC=[0,1,0,-1];
const typ=s=>LAYOUT[(s/C)|0][s%C];
const isWall=s=>typ(s)==='#', isTerm=s=>typ(s)==='G'||typ(s)==='L';
const rc=s=>[(s/C)|0,s%C], lbl=s=>`(${(s/C)|0},${s%C})`;
const ALL=[...Array(N).keys()], NT=ALL.filter(s=>!isWall(s)&&!isTerm(s));
const P={slip:0.1,gamma:0.9,temp:1,speed:1,horizon:10,sigma:0.5};
function move(s,a){const[r,c]=rc(s),nr=r+DR[a],nc=c+DC[a];if(nr<0||nr>=R||nc<0||nc>=C)return s;const s2=nr*C+nc;return isWall(s2)?s:s2}
const R_GOAL=10,R_LAVA=-50,R_STEP=-1;
const reward=s2=>typ(s2)==='G'?R_GOAL:typ(s2)==='L'?R_LAVA:R_STEP;
function trans(s,a){const m=new Map(),add=(b,p)=>{if(p<=0)return;const s2=move(s,b);m.set(s2,(m.get(s2)||0)+p)};
  add(a,1-P.slip);add((a+1)%4,P.slip/2);add((a+3)%4,P.slip/2);return[...m].map(([s2,p])=>({s2,p,r:reward(s2)}))}
function sampleMove(s,a){const u=rand();const act=u<1-P.slip?a:u<1-P.slip/2?(a+1)%4:(a+3)%4;return{act,s2:move(s,act)}}
function sampleCat(p){let u=rand();for(let i=0;i<p.length;i++){u-=p[i];if(u<=0)return i}return p.length-1}
const backup=(V,s,a)=>trans(s,a).reduce((q,{s2,p,r})=>q+p*(r+P.gamma*(isTerm(s2)?0:V[s2])),0);
function evalSweep(V,pi){const nv=V.slice();let d=0;for(const s of NT){let v=0;for(let a=0;a<4;a++)if(pi[s][a])v+=pi[s][a]*backup(V,s,a);d=Math.max(d,Math.abs(v-V[s]));nv[s]=v}return{V:nv,delta:d}}
function viSweep(V){const nv=V.slice();let d=0;for(const s of NT){const v=Math.max(...[0,1,2,3].map(a=>backup(V,s,a)));d=Math.max(d,Math.abs(v-V[s]));nv[s]=v}return{V:nv,delta:d}}
function converge(f){let V=new Array(N).fill(0);for(let i=0;i<20000;i++){const o=f(V);V=o.V;if(o.delta<1e-8)break}return V}
const qFrom=V=>ALL.map(s=>NT.includes(s)?[0,1,2,3].map(a=>backup(V,s,a)):[0,0,0,0]);
const argmax=q=>{let b=0;for(let a=1;a<4;a++)if(q[a]>q[b]+1e-9)b=a;return b};
let M=null;
function model(){
  if(M)return M;
  const Vs=converge(viSweep),Qs=qFrom(Vs);
  const pi={random:ALL.map(()=>[.25,.25,.25,.25]),
    greedy:ALL.map(s=>{const p=[0,0,0,0];p[argmax(Qs[s])]=1;return p}),
    soft:ALL.map(s=>{const q=Qs[s],m=Math.max(...q),e=q.map(x=>Math.exp((x-m)/P.temp)),z=e.reduce((a,b)=>a+b);return e.map(x=>x/z)})};
  return M={Vs,Qs,pi,Vpi:{},Qpi:{}};
}
function Vpi(k){const m=model();if(!m.Vpi[k]){m.Vpi[k]=converge(V=>evalSweep(V,m.pi[k]));m.Qpi[k]=qFrom(m.Vpi[k])}return m.Vpi[k]}
function Qpi(k){Vpi(k);return M.Qpi[k]}
function invalidate(){M=null;active&&active.onModel&&active.onModel()}
const fmt=(v,d=1)=>{if(Math.abs(v)<.5*10**-d)v=0;return(v<0?'−':'')+Math.abs(v).toFixed(d)};
const sgn=v=>v>0?'pos':v<0?'neg':'';
const PNAME={random:'Random',soft:'Softmax',greedy:'Greedy'};

/* ================= Agent runner ================= */
const MOVE=.5,GAP=.25;
class Runner{
  constructor(o){this.o=o;this.reset()}
  reset(){Object.assign(this,{s:S0,t:0,steps:[],from:S0,to:S0,prog:1,wait:.5,done:false,trail:[S0],bump:-1,age:99})}
  step(){
    if(this.done){this.o.onEnd&&this.o.onEnd(this);this.reset();return}
    const a=sampleCat(this.o.pi()[this.s]);const{act,s2}=sampleMove(this.s,a);const r=reward(s2);
    const st={t:this.t,s:this.s,a,act,s2,r};this.steps.push(st);
    Object.assign(this,{from:this.s,to:s2,prog:0,age:0,bump:s2===this.s?act:-1,s:s2});this.t++;this.trail.push(s2);
    this.o.onStep&&this.o.onStep(this,st);
    const H=this.o.horizon?this.o.horizon():120;
    if(isTerm(s2)||this.t>=H)this.done=true;
    this.wait=this.done?(this.o.hold||1.2):GAP;
  }
  update(dt,play){this.age+=dt*P.speed;if(this.prog<1){this.prog=Math.min(1,this.prog+dt*P.speed/MOVE);return}
    if(!play)return;this.wait-=dt*P.speed;if(this.wait<=0)this.step()}
  manual(){this.prog=1;this.step()}
  pos(){const e=1-Math.pow(1-this.prog,3),[r1,c1]=rc(this.from),[r2,c2]=rc(this.to);let r=r1+(r2-r1)*e,c=c1+(c2-c1)*e;
    if(this.bump>=0){const k=Math.sin(Math.PI*this.prog)*.22;r+=DR[this.bump]*k;c+=DC[this.bump]*k}return{r,c}}
}

/* ================= Canvas ================= */
let g=null,W=0,H=0,geo={ox:0,oy:0,cs:1},tk={};
/* Point all drawing at one canvas context of CSS size w×h (called by the layout before every draw). */
export function bindCanvas(ctx,w,h){g=ctx;W=w;H=h;const cs=Math.floor(Math.min(W,H)/5.05);geo={cs,ox:Math.round((W-cs*C)/2),oy:Math.round((H-cs*R)/2)}}
export const getGeo=()=>geo;
export function readTokens(){const cs=getComputedStyle(document.documentElement),o={};for(const k of['bg','surface','ink','muted','line','cell','wall','agent','pos','neg','hl'])o[k]=cs.getPropertyValue('--'+k).trim();tk=o}
const rgba=(h,a)=>{h=h.replace('#','');if(h.length===3)h=[...h].map(x=>x+x).join('');const n=parseInt(h,16);return`rgba(${n>>16&255},${n>>8&255},${n&255},${a})`};
const center=s=>{const[r,c]=rc(s);return{x:geo.ox+(c+.5)*geo.cs,y:geo.oy+(r+.5)*geo.cs}};
const toXY=(r,c)=>({x:geo.ox+(c+.5)*geo.cs,y:geo.oy+(r+.5)*geo.cs});
function rr(x,y,w,h,r){g.beginPath();g.roundRect?g.roundRect(x,y,w,h,r):g.rect(x,y,w,h)}
const font=(w,sz,f='mono')=>`${w} ${Math.round(sz)}px ${f==='mono'?'"IBM Plex Mono",ui-monospace,monospace':'"IBM Plex Sans",system-ui,sans-serif'}`;
function text(t,x,y,col,sz,w=600,f='mono',al='center'){g.fillStyle=col;g.font=font(w,sz,f);g.textAlign=al;g.textBaseline='middle';g.fillText(t,x,y)}
function arrow(x1,y1,x2,y2,w,col){const a=Math.atan2(y2-y1,x2-x1),h=Math.max(6,w*2.6);g.strokeStyle=col;g.fillStyle=col;g.lineWidth=w;g.lineCap='round';
  g.beginPath();g.moveTo(x1,y1);g.lineTo(x2-Math.cos(a)*h*.7,y2-Math.sin(a)*h*.7);g.stroke();
  g.beginPath();g.moveTo(x2,y2);g.lineTo(x2-h*Math.cos(a-.45),y2-h*Math.sin(a-.45));g.lineTo(x2-h*Math.cos(a+.45),y2-h*Math.sin(a+.45));g.closePath();g.fill()}
function heatCol(v,scale){return rgba(v>=0?tk.pos:tk.neg,Math.min(1,Math.abs(v)/scale)*.62)}

function drawWorld(o={}){
  const{ox,oy,cs}=geo,pad=Math.max(1.5,cs*.025);
  const hv=o.heat?NT.map(s=>o.heat[s]):[0],hsP=o.heatScale||Math.max(5,...hv),hsN=o.heatScale||Math.max(5,...hv.map(v=>-v));
  g.clearRect(0,0,W,H);
  for(const s of ALL){const[r,c]=rc(s),x=ox+c*cs,y=oy+r*cs,t=typ(s);
    rr(x+pad,y+pad,cs-2*pad,cs-2*pad,cs*.07);
    if(t==='#'){g.fillStyle=tk.wall;g.fill();g.save();g.clip();g.strokeStyle=rgba(tk.bg,.18);g.lineWidth=2;
      for(let k=-cs;k<cs;k+=cs/6){g.beginPath();g.moveTo(x+k,y+cs);g.lineTo(x+k+cs,y);g.stroke()}g.restore();continue}
    if(t==='G'||t==='L'){const col=t==='G'?tk.pos:tk.neg;g.fillStyle=tk.cell;g.fill();g.fillStyle=rgba(col,.22);g.fill();g.strokeStyle=col;g.lineWidth=2.5;g.stroke();
      text(t==='G'?'+'+R_GOAL:fmt(R_LAVA,0),x+cs/2,y+cs*.44,col,cs*.22);text(t==='G'?'goal':'lava',x+cs/2,y+cs*.7,col,cs*.11,500,'sans');continue}
    g.fillStyle=tk.cell;g.fill();
    if(o.heat){const v=o.heat[s];g.fillStyle=heatCol(v,v>=0?hsP:hsN);g.fill()}
    if(o.visits){g.fillStyle=rgba(tk.agent,Math.sqrt(o.visits[s]/Math.max(1,o.vmax))*.6);g.fill()}
    if(s===S0)text('start',x+cs*.07,y+cs*.89,tk.muted,cs*.1,500,'sans','left');
  }
  if(o.q){const qv=NT.flatMap(s=>o.q[s]),scP=o.qScale||Math.max(5,...qv),scN=o.qScale||Math.max(5,...qv.map(v=>-v));for(const s of NT){const{x,y}=center(s),h=cs/2-pad;
    const pts=[[[-h,-h],[h,-h]],[[h,-h],[h,h]],[[h,h],[-h,h]],[[-h,h],[-h,-h]]];
    for(let a=0;a<4;a++){const v=o.q[s][a];g.beginPath();g.moveTo(x,y);g.lineTo(x+pts[a][0][0],y+pts[a][0][1]);g.lineTo(x+pts[a][1][0],y+pts[a][1][1]);g.closePath();
      g.fillStyle=rgba(v>=0?tk.pos:tk.neg,.08+Math.min(1,Math.abs(v)/(v>=0?scP:scN))*.62);g.fill();g.strokeStyle=rgba(tk.surface,.9);g.lineWidth=1.2;g.stroke();
      text(fmt(v),x+DC[a]*cs*.3,y+DR[a]*cs*.31,tk.ink,cs*.105,600)}
    if(o.qMark){const b=argmax(o.q[s]);g.beginPath();g.arc(x+DC[b]*cs*.3,y+DR[b]*cs*.31,cs*.115,0,7);g.strokeStyle=tk.ink;g.lineWidth=1.5;g.stroke()}}}
  if(o.nums){for(const s of NT){const{x,y}=center(s);text(fmt(o.nums[s],o.numDigits??1),x,o.pi?y+cs*.33:y,tk.ink,o.pi?cs*.13:cs*.19)}}
  if(o.pi){for(const s of NT){const{x,y}=center(s),p=o.pi[s],cy=o.nums?y-cs*.06:y,L=o.nums?cs*.26:cs*.36;
    for(let a=0;a<4;a++){if(p[a]<.02)continue;const len=L*(.35+.65*p[a]);
      arrow(x+DC[a]*cs*.05,cy+DR[a]*cs*.05,x+DC[a]*len,cy+DR[a]*len,1.5+3.5*p[a],rgba(tk.ink,.25+.65*p[a]))}}}
  if(o.trails)o.trails.forEach((tr,i)=>{const off=((i%5)-2)*cs*.035;g.strokeStyle=rgba(o.trailCol||tk.agent,o.trailAlpha?o.trailAlpha(i):.55);g.lineWidth=Math.max(2,cs*.03);g.lineJoin='round';
    g.beginPath();tr.forEach((s,j)=>{const{x,y}=center(s);j?g.lineTo(x+off,y+off):g.moveTo(x+off,y+off)});g.stroke()});
  if(o.sel!=null){const[r,c]=rc(o.sel);rr(ox+c*cs+pad,oy+r*cs+pad,cs-2*pad,cs-2*pad,cs*.07);g.strokeStyle=tk.hl;g.lineWidth=4;g.stroke()}
  if(o.fog!=null){const[ar,ac]=rc(o.fog);for(const s of ALL){const[r,c]=rc(s);if(Math.abs(r-ar)<=1&&Math.abs(c-ac)<=1)continue;
    rr(ox+c*cs+1,oy+r*cs+1,cs-2,cs-2,cs*.07);g.fillStyle=rgba(tk.surface,.94);g.fill();text('?',ox+(c+.5)*cs,oy+(r+.5)*cs,rgba(tk.muted,.6),cs*.25)}
    const x=ox+(ac-1)*cs,y=oy+(ar-1)*cs;g.setLineDash([8,6]);g.strokeStyle=tk.agent;g.lineWidth=2;g.strokeRect(x,y,cs*3,cs*3);g.setLineDash([])}
}
function drawAgent(p){const{x,y}=toXY(p.r,p.c),rad=geo.cs*.2;
  g.beginPath();g.arc(x,y+rad*.15,rad*1.05,0,7);g.fillStyle=rgba('#000000',.18);g.fill();
  g.beginPath();g.arc(x,y,rad,0,7);g.fillStyle=tk.agent;g.fill();g.lineWidth=3;g.strokeStyle=tk.surface;g.stroke();
  g.beginPath();g.arc(x-rad*.3,y-rad*.3,rad*.28,0,7);g.fillStyle=rgba('#ffffff',.45);g.fill()}
let floats=[];
function addFloat(s,t,col){const{x,y}=center(s);floats.push({x,y,t,col,life:1})}
function drawFloats(dt){floats=floats.filter(f=>(f.life-=dt*P.speed*.9)>0);for(const f of floats){g.globalAlpha=Math.min(1,f.life*1.5);
  text(f.t,f.x,f.y-geo.cs*.35-(1-f.life)*geo.cs*.5,tk[f.col]||tk.ink,geo.cs*.22,700);g.globalAlpha=1}}
function drawTrail(run){const tr=run.trail.slice(0,-1);if(!tr.length)return;const p=run.pos(),q=toXY(p.r,p.c);
  g.strokeStyle=rgba(tk.agent,.45);g.lineWidth=Math.max(2,geo.cs*.035);g.lineJoin='round';g.lineCap='round';g.beginPath();
  tr.forEach((s,j)=>{const{x,y}=center(s);j?g.lineTo(x,y):g.moveTo(x,y)});g.lineTo(q.x,q.y);g.stroke()}

function drawTrans(sel,ai){const c=center(sel),cs=geo.cs;for(const{s2,p}of trans(sel,ai)){
    if(s2===sel){g.beginPath();g.arc(c.x+DC[ai]*cs*.32,c.y+DR[ai]*cs*.32,cs*.12,0,7);g.strokeStyle=tk.hl;g.lineWidth=3;g.stroke();
      text(p.toFixed(2),c.x+DC[ai]*cs*.32,c.y+DR[ai]*cs*.32,tk.ink,cs*.1);continue}
    const d=center(s2),mx=c.x+(d.x-c.x)*.36,my=c.y+(d.y-c.y)*.36;arrow(c.x,c.y,d.x-(d.x-c.x)*.28,d.y-(d.y-c.y)*.28,2+5*p,tk.hl);
    const tw=cs*.3;rr(mx-tw/2,my-cs*.09,tw,cs*.18,cs*.05);g.fillStyle=tk.surface;g.fill();text(p.toFixed(2),mx,my,tk.ink,cs*.12)}
  text(ARW[ai]+' intended',c.x,c.y-cs*.33,tk.ink,cs*.1,600,'sans')}

/* Continuous-action agent (Gaussian policy) */
const gauss=()=>{let u=0,v=0;while(!u)u=rand();while(!v)v=rand();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)};
const clip=v=>Math.max(-1,Math.min(1,v));
const cellOf=(x,y)=>{const c=Math.floor(x),r=Math.floor(y);return r<0||r>=R||c<0||c>=C?-1:r*C+c};
function makeCont(){const ct={};
  ct.reset=()=>Object.assign(ct,{x:.5,y:4.5,fx:.5,fy:4.5,prog:1,wait:.4,done:false,samples:[],a:[0,0],raw:[0,0],z:[0,0],mu:[0,-1],t:0,bump:false,stepped:false});
  ct.step=()=>{if(ct.done){ct.reset();return}const s=cellOf(ct.x,ct.y),ga=argmax(model().Qs[s]);ct.mu=[DC[ga],DR[ga]];
    const one=()=>{const z=[gauss(),gauss()];return{z,raw:[ct.mu[0]+P.sigma*z[0],ct.mu[1]+P.sigma*z[1]]}};
    const f=one();ct.z=f.z;ct.raw=f.raw;ct.a=f.raw.map(clip);ct.samples=Array.from({length:29},()=>one().raw.map(clip));ct.stepped=true;
    let nx=ct.x+.45*ct.a[0],ny=ct.y+.45*ct.a[1];const s2=cellOf(nx,ny);ct.bump=s2<0||isWall(s2);if(ct.bump){nx=ct.x;ny=ct.y}
    Object.assign(ct,{fx:ct.x,fy:ct.y,x:nx,y:ny,prog:0});ct.t++;if((!ct.bump&&isTerm(s2))||ct.t>150)ct.done=true;ct.wait=ct.done?1.1:GAP*.4};
  ct.update=(dt,play)=>{if(ct.prog<1){ct.prog=Math.min(1,ct.prog+dt*P.speed/(MOVE*.6));return}if(!play)return;ct.wait-=dt*P.speed;if(ct.wait<=0)ct.step()};
  ct.manual=()=>{ct.prog=1;ct.step()};
  ct.draw=()=>{const e=1-Math.pow(1-ct.prog,3),x=ct.fx+(ct.x-ct.fx)*e,y=ct.fy+(ct.y-ct.fy)*e,L=geo.cs*.9,bx=geo.ox+ct.fx*geo.cs,by=geo.oy+ct.fy*geo.cs;
    for(const s of ct.samples)arrow(bx,by,bx+s[0]*L,by+s[1]*L,1.5,rgba(tk.agent,.22));
    if(ct.stepped)arrow(bx,by,bx+ct.a[0]*L,by+ct.a[1]*L,3.5,tk.hl);
    drawAgent({r:y-.5,c:x-.5})};
  ct.logp=()=>{let q=0;for(let i=0;i<2;i++)q+=(ct.raw[i]-ct.mu[i])**2/P.sigma**2+2*Math.log(P.sigma);return-.5*(q+2*Math.log(2*Math.PI))};
  ct.reset();return ct}

/* ================= Learners (Part 2) ================= */
const CAP=100;
const softmax=th=>{const m=Math.max(...th),e=th.map(x=>Math.exp(x-m)),z=e.reduce((a,b)=>a+b);return e.map(x=>x/z)};
const zeros4=()=>ALL.map(()=>[0,0,0,0]);
const greedyOf=Q=>ALL.map(s=>{const p=[0,0,0,0];p[argmax(Q[s])]=1;return p});
const L_Q=(o={})=>({alpha:.5,eps:.1,replay:!!o.replay,init(){this.Q=zeros4();this.buf=[];this.last=null},
  act(s){return rand()<this.eps?(rand()*4)|0:argmax(this.Q[s])},
  upd(s,a,r,s2,term,al){const tgt=r+(term?0:P.gamma*Math.max(...this.Q[s2]));this.Q[s][a]+=al*(tgt-this.Q[s][a])},
  observe(s,a,r,s2,term){this.upd(s,a,r,s2,term,this.alpha);this.last=[s,a];if(this.replay){this.buf.push([s,a,r,s2,term]);if(this.buf.length>5000)this.buf.shift();
    for(let k=0;k<10;k++){const x=this.buf[(rand()*this.buf.length)|0];this.upd(...x,this.alpha*.5)}}},
  pi(){return greedyOf(this.Q)},V(){return this.Q.map(q=>Math.max(...q))}});
const L_PG=()=>({alpha:.1,beta:.1,init(){this.th=zeros4();this.Vb=new Array(N).fill(0);this.ep=[]},
  act(s){return sampleCat(softmax(this.th[s]))},observe(s,a,r){this.ep.push([s,a,r])},
  endEpisode(){let G=0;for(let i=this.ep.length-1;i>=0;i--){const[s,a,r]=this.ep[i];G=r+P.gamma*G;const A=G-this.Vb[s];this.Vb[s]+=this.beta*(G-this.Vb[s]);
    const p=softmax(this.th[s]);for(let b=0;b<4;b++)this.th[s][b]+=this.alpha*A*((b===a)-p[b])}this.ep=[]},
  pi(){return ALL.map(s=>softmax(this.th[s]))},V(){return this.Vb}});
const L_AC=()=>({alpha:.1,beta:.2,init(){this.th=zeros4();this.Vc=new Array(N).fill(0);this.delta=0},
  act(s){return sampleCat(softmax(this.th[s]))},
  observe(s,a,r,s2,term){const d=r+(term?0:P.gamma*this.Vc[s2])-this.Vc[s];this.delta=d;this.Vc[s]+=this.beta*d;const p=softmax(this.th[s]);for(let b=0;b<4;b++)this.th[s][b]+=this.alpha*d*((b===a)-p[b])},
  pi(){return ALL.map(s=>softmax(this.th[s]))},V(){return this.Vc}});
const L_Model=()=>({eps:.05,init(){this.cnt=ALL.map(()=>[0,1,2,3].map(()=>new Map()));this.tot=zeros4();this.Vm=new Array(N).fill(0);this.Qm=zeros4()},
  model(s,a){const t=this.tot[s][a];if(!t)return[{s2:s,p:1,r:0}];return[...this.cnt[s][a]].map(([s2,c])=>({s2,p:c/t,r:reward(s2)}))},
  plan(sweeps=6){for(let k=0;k<sweeps;k++){const nv=this.Vm.slice();for(const s of NT){for(let a=0;a<4;a++)this.Qm[s][a]=this.model(s,a).reduce((q,{s2,p,r})=>q+p*(r+P.gamma*(isTerm(s2)?0:this.Vm[s2])),0);nv[s]=Math.max(...this.Qm[s])}this.Vm=nv}},
  act(s){return rand()<this.eps?(rand()*4)|0:argmax(this.Qm[s])},
  observe(s,a,r,s2){const m=this.cnt[s][a];m.set(s2,(m.get(s2)||0)+1);this.tot[s][a]++;this.plan()},
  known(){return NT.reduce((k,s)=>k+this.tot[s].filter(x=>x>0).length,0)},
  err(){let e=0,n=0;for(const s of NT)for(let a=0;a<4;a++){if(!this.tot[s][a])continue;n++;const tp=new Map(trans(s,a).map(x=>[x.s2,x.p]));const keys=new Set([...tp.keys(),...this.cnt[s][a].keys()]);
    let tv=0;for(const k of keys)tv+=Math.abs((tp.get(k)||0)-(this.cnt[s][a].get(k)||0)/this.tot[s][a]);e+=tv/2}return n?e/n:null},
  pi(){return greedyOf(this.Qm)},V(){return this.Vm}});
class Trainer{
  constructor(mk){this.mk=mk;this.reset()}
  reset(){this.L=this.mk();this.L.init();Object.assign(this,{s:S0,t:0,G:0,d:1,steps:0,episodes:0,curve:[],trail:[S0],from:S0,to:S0,prog:1,wait:.3,bump:-1})}
  envStep(){const s=this.s,a=this.L.act(s),{act,s2}=sampleMove(s,a),r=reward(s2),term=isTerm(s2);
    this.L.observe(s,a,r,s2,term);this.G+=this.d*r;this.d*=P.gamma;this.t++;this.steps++;
    Object.assign(this,{from:s,to:s2,bump:s2===s?act:-1,s:s2});this.trail.push(s2);
    if(term||this.t>=CAP){this.L.endEpisode&&this.L.endEpisode();this.curve.push({x:this.steps,y:this.G});this.episodes++;
      Object.assign(this,{s:S0,t:0,G:0,d:1,trail:[S0],from:S0,to:S0,bump:-1});return true}return false}
  update(dt,play,fast){if(fast){if(play){const n=Math.round(30*P.speed);for(let i=0;i<n;i++)this.envStep()}this.prog=1;return}
    if(this.prog<1){this.prog=Math.min(1,this.prog+dt*P.speed/(MOVE*.6));return}if(!play)return;this.wait-=dt*P.speed;
    if(this.wait<=0){const ended=this.envStep();this.prog=ended?1:0;this.wait=ended?.6:GAP*.5}}
  manual(){const e=this.envStep();this.prog=e?1:0}
  pos(){const e=1-Math.pow(1-this.prog,3),[r1,c1]=rc(this.from),[r2,c2]=rc(this.to);let r=r1+(r2-r1)*e,c=c1+(c2-c1)*e;
    if(this.bump>=0){const k=Math.sin(Math.PI*this.prog)*.22;r+=DR[this.bump]*k;c+=DC[this.bump]*k}return{r,c}}
  avg(n=30){const c=this.curve.slice(-n);return c.length?c.reduce((a,b)=>a+b.y,0)/c.length:null}
}
/* Learning-curve chart: moving average of episode return vs environment steps */
function chart(c,w,h,series,o={}){const lo=o.lo??-12,hi=o.hi??0,win=o.win||30,fs=o.fs||12,pl=fs*3.4,pr=o.pr??14,pt=fs*1.2,pb=fs*2.4;
  const xMax=o.xMax||Math.max(2000,...series.map(s=>s.c.length?s.c.at(-1).x:0));
  const X=x=>pl+x/xMax*(w-pl-pr),Y=y=>pt+(hi-Math.max(lo,Math.min(hi,y)))/(hi-lo)*(h-pt-pb);
  c.clearRect(0,0,w,h);c.font=font(400,fs);c.textBaseline='middle';
  for(const v of o.ticks||[-12,-8,-4,0]){c.strokeStyle=rgba(tk.ink,.08);c.lineWidth=1;c.beginPath();c.moveTo(pl,Y(v));c.lineTo(w-pr,Y(v));c.stroke();c.fillStyle=tk.muted;c.textAlign='right';c.fillText(fmt(v,0),pl-6,Y(v))}
  c.textBaseline='top';c.fillStyle=tk.muted;c.textAlign='left';c.fillText('0',pl,h-pb+fs*.5);c.textAlign='right';c.fillText(xMax>=1000?(Math.round(xMax/100)/10)+'k':String(xMax),w-pr,h-pb+fs*.5);c.textAlign='center';
  c.fillText(o.xlabel||'environment steps',(pl+w-pr)/2,h-pb+fs*.5);
  const J=o.ref??model().Vs[S0];c.setLineDash([6,5]);c.strokeStyle=tk.pos;c.lineWidth=1.5;c.beginPath();c.moveTo(pl,Y(J));c.lineTo(w-pr,Y(J));c.stroke();c.setLineDash([]);
  c.fillStyle=tk.pos;c.textAlign='left';c.textBaseline='bottom';c.fillText('optimal',pl+4,Y(J)-3);
  for(const s of series){if(!s.c.length)continue;c.strokeStyle=s.col;c.lineWidth=o.lw||2.5;c.lineJoin='round';c.beginPath();let sum=0;
    s.c.forEach((p,i)=>{sum+=p.y;if(i>=win)sum-=s.c[i-win].y;const m=sum/Math.min(i+1,win);i?c.lineTo(X(p.x),Y(m)):c.moveTo(X(p.x),Y(m))});c.stroke();
    if(s.label){const p=s.c.at(-1);let m=0,k=Math.min(win,s.c.length);for(let i=s.c.length-k;i<s.c.length;i++)m+=s.c[i].y;m/=k;c.fillStyle=s.col;c.font=font(600,fs);const lx=X(p.x),left=lx<pl+c.measureText(s.label).width+10;c.textAlign=left?'left':'right';c.textBaseline='bottom';c.fillText(s.label,left?lx+8:lx,Y(m)-6)}}}
const statsHtml=tr=>{const a=tr.avg();return`<div class="stats"><div><div class="muted">env steps</div><div class="big">${tr.steps}</div></div><div><div class="muted">episodes</div><div class="big">${tr.episodes}</div></div><div><div class="muted">avg return</div><div class="big ${a==null?'':sgn(a)}">${a==null?'—':fmt(a,2)}</div></div></div>`};
function trainerScene(o){let fast=o.fast??true;const tr=new Trainer(o.mk);
  return{controls:()=>[{type:'seg',label:'Animation',opts:[['watch','Every step'],['fast','Fast-forward']],get:()=>fast?'fast':'watch',set:v=>fast=v==='fast'},...(o.controls?o.controls(tr):[])],
    onModel(){tr.reset()},enter(){tr.reset()},reset(){tr.reset()},step(){tr.manual()},update(dt){tr.update(dt,playing,fast)},
    draw(){o.draw(tr,fast);drawTrail(tr);drawAgent(tr.pos())},
    panelCanvas:o.curve!==false,drawPanel(pg,w,h){chart(pg,w,h,[{c:tr.curve,col:tk.agent}])},
    live(){return o.live?o.live(tr):statsHtml(tr)}}}

/* ================= Part 3: batch policy gradient on a table of logits ================= */
const T3=50;
const undisc=f=>{const g0=P.gamma;P.gamma=1;try{return f()}finally{P.gamma=g0}};
const V1pi=pi=>undisc(()=>converge(V=>evalSweep(V,pi)));
function V1star(){const m=model();if(!m.V1s)m.V1s=undisc(()=>converge(viSweep));return m.V1s}
function V1soft(){const m=model();if(!m.V1soft)m.V1soft=V1pi(m.pi.soft);return m.V1soft}
const epReturn=ep=>ep.reduce((x,e)=>x+e.r,0);
const rewardToGo=ep=>{let g=0;const out=new Array(ep.length);for(let i=ep.length-1;i>=0;i--){g+=ep[i].r;out[i]=g}return out};
function makePG3(kind,o={}){const pg={kind,alpha:o.alpha??.01,batch:o.batch??10,beta:.1,
  init(){this.th=zeros4();this.Vb=new Array(N).fill(0);this.epoch=0;this.curve=[]},
  pi(s){return softmax(this.th[s])},table(){return ALL.map(s=>this.pi(s))},
  rollout(){let s=S0;const ep=[];for(let t=0;t<T3&&!isTerm(s);t++){const a=sampleCat(this.pi(s)),{act,s2}=sampleMove(s,a);ep.push({t,s,a,act,s2,r:reward(s2)});s=s2}return ep},
  weights(ep){const R=epReturn(ep),rtg=rewardToGo(ep);return ep.map((e,i)=>this.kind==='R'?R:this.kind==='rtg'?rtg[i]:rtg[i]-this.Vb[e.s])},
  update(eps){const grad=zeros4();for(const ep of eps){const w=this.weights(ep);ep.forEach((e,i)=>{const p=this.pi(e.s);for(let b=0;b<4;b++)grad[e.s][b]+=w[i]*((b===e.a)-p[b])})}
    for(const s of NT)for(let b=0;b<4;b++)this.th[s][b]+=this.alpha*grad[s][b]/eps.length;
    if(this.kind==='base')for(const ep of eps){const rtg=rewardToGo(ep);ep.forEach((e,i)=>this.Vb[e.s]+=this.beta*(rtg[i]-this.Vb[e.s]))}
    this.epoch++;const avg=eps.reduce((x,ep)=>x+epReturn(ep),0)/eps.length;this.curve.push({x:this.epoch,y:avg});return avg},
  trainEpoch(){const eps=[];for(let i=0;i<this.batch;i++)eps.push(this.rollout());return this.update(eps)}};
  pg.init();return pg}
function drawEpisode(ep,col){if(!ep.length)return;g.strokeStyle=col;g.lineWidth=Math.max(2,geo.cs*.03);g.lineJoin='round';g.beginPath();
  const a=center(ep[0].s);g.moveTo(a.x,a.y);for(const e of ep){const b=center(e.s2);g.lineTo(b.x,b.y)}g.stroke()}
function pill(x,y,t,col){const cs=geo.cs;g.font=font(600,cs*.11);const w=g.measureText(t).width+cs*.08;rr(x-w/2,y-cs*.08,w,cs*.16,cs*.04);g.fillStyle=tk.surface;g.fill();g.strokeStyle=col;g.lineWidth=1.5;g.stroke();text(t,x,y,col,cs*.11)}
const hlCode=l=>l.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/(#.*)$/,'<span class="cm">$1</span>').replace(/\b(def|return)\b/g,'<span class="kw">$1</span>');

/* ================= Slides ================= */
const policyLive=(pi,s,hi)=>`<div class="bars">${[0,1,2,3].map(a=>`<span class="${a===hi?'on':''}">${ARW[a]}</span><div class="bar"><i style="width:${pi[s][a]*100}%"></i></div><span class="${a===hi?'on':''}">${pi[s][a].toFixed(2)}</span>`).join('')}</div>`;
const kindSeg=(get,set,opts=['random','soft','greedy'])=>({type:'seg',label:'Policy π',opts:opts.map(k=>[k,PNAME[k]]),get,set});
const slipCtl={type:'range',label:'Slip',min:0,max:.6,step:.05,get:()=>P.slip,set:v=>{P.slip=v;invalidate()},fmt:v=>v.toFixed(2)};
const gammaCtl={type:'range',label:'γ',min:0,max:.99,step:.01,get:()=>P.gamma,set:v=>{P.gamma=v;invalidate()},fmt:v=>v.toFixed(2)};
const tempCtl={type:'range',label:'Temperature',min:.1,max:5,step:.1,get:()=>P.temp,set:v=>{P.temp=v;invalidate()},fmt:v=>v.toFixed(1)};
const EQ=s=>'='+s;
const scenes=[];

/* Cover */
{const run=new Runner({pi:()=>model().pi.soft});
scenes.push({controls:()=>[],enter(){run.reset()},update:dt=>run.update(dt,playing),step:()=>run.manual(),reset:()=>run.reset(),
  draw(){drawWorld();drawTrail(run);drawAgent(run.pos())}})}

/* Section header */
function header(kind){const run=new Runner({pi:()=>model().pi[kind]});
  return{controls:()=>[],enter(){run.reset()},update:dt=>run.update(dt,playing),step:()=>run.manual(),reset:()=>run.reset(),
    draw(){drawWorld();drawTrail(run);drawAgent(run.pos())}}}
scenes.push(header('random'));

/* What can RL do */
scenes.push(trainerScene({mk:L_AC,curve:false,
  draw(tr){drawWorld({pi:tr.L.pi()})},
  live(tr){const a=tr.avg();return`<div class="muted">The robot on the right started knowing nothing. It learns only from reward.</div>${statsHtml(tr)}`}}));

/* Meet the gridworld */
{const run=new Runner({pi:()=>model().pi.soft,onStep:(r,st)=>addFloat(st.s2,(st.r>0?'+':'')+fmt(st.r,0),st.r>0?'pos':'neg')});
scenes.push({controls:()=>[slipCtl],enter(){run.reset()},update:dt=>run.update(dt,playing),step:()=>run.manual(),reset:()=>run.reset(),
  draw(dt){drawWorld();drawTrail(run);drawAgent(run.pos());drawFloats(dt)}})}

/* 1 — loop */
{const run=new Runner({pi:()=>model().pi.random});
scenes.push({extra:`<svg class="loop" viewBox="0 0 320 190" role="img" aria-label="Agent–environment loop">
    <rect class="box" x="100" y="10" width="120" height="44" rx="8"/><text x="160" y="38" text-anchor="middle">Agent</text>
    <rect class="box" x="85" y="136" width="150" height="44" rx="8"/><text x="160" y="164" text-anchor="middle">Environment</text>
    <g class="flow" id="fa"><path d="M220,32 C300,32 300,158 241,158"/><polygon points="235,158 245,152 245,164"/></g>
    <g class="flow" id="fo"><path d="M85,158 C20,158 20,32 84,32"/><polygon points="94,32 84,26 84,38"/></g>
    <text x="303" y="100" text-anchor="middle" font-style="italic">a<tspan baseline-shift="sub" font-size="13">t</tspan></text>
    <text x="22" y="90" text-anchor="middle" font-style="italic">s<tspan baseline-shift="sub" font-size="13">t+1</tspan></text>
    <text x="22" y="116" text-anchor="middle" font-style="italic">r<tspan baseline-shift="sub" font-size="13">t</tspan></text></svg>`,
  controls:()=>[],enter(){run.reset()},update:dt=>run.update(dt,playing),step:()=>run.manual(),reset:()=>run.reset(),
  draw(){drawWorld();drawTrail(run);drawAgent(run.pos())},
  dom(){const age=run.steps.length?run.age:99;document.getElementById('fa')?.classList.toggle('on',age<MOVE*.5);document.getElementById('fo')?.classList.toggle('on',age>=MOVE*.5&&age<MOVE*1.3)},
  live(){const st=run.steps.at(-1);return st?`<div class="mono">t=${st.t} &nbsp; s=${lbl(st.s)} &nbsp; a=${ARW[st.a]} &nbsp; r=<span class="${sgn(st.r)}">${fmt(st.r,0)}</span> &nbsp; s′=${lbl(st.s2)}</div>`:''}})}

/* 2 — states & observations */
{const run=new Runner({pi:()=>model().pi.random});
scenes.push({controls:()=>[],enter(){run.reset()},update:dt=>run.update(dt,playing),step:()=>run.manual(),reset:()=>run.reset(),
  draw(){drawWorld({});drawAgent(run.pos())},
  live(){return`<div><i>s<sub>t</sub></i> = <span class="big">${lbl(run.to)}</span> <span class="muted">(row, column)</span></div>`}})}

/* 3 — action spaces */
{const run=new Runner({pi:()=>model().pi.random});let mode='disc';const ct=makeCont();
scenes.push({controls:()=>[{type:'seg',label:'Action space',opts:[['disc','Discrete'],['cont','Continuous']],get:()=>mode,set:v=>{mode=v;run.reset();ct.reset()}},
    ...(mode==='cont'?[{type:'range',label:'Spread σ',min:.05,max:1.5,step:.05,get:()=>P.sigma,set:v=>P.sigma=v,fmt:v=>v.toFixed(2)}]:[])],
  enter(){run.reset();ct.reset()},
  update(dt){mode==='disc'?run.update(dt,playing):ct.update(dt,playing)},
  step(){mode==='disc'?run.manual():ct.manual()},reset(){run.reset();ct.reset()},
  draw(){drawWorld();mode==='disc'?drawAgent(run.pos()):ct.draw()},
  live(){if(mode==='disc'){const st=run.steps.at(-1),a=st&&run.prog<1?st.a:-1,k=i=>`<div class="k ${a===i?'on':''}">${ARW[i]}</div>`;
      return`<div class="compass"><div></div>${k(0)}<div></div>${k(3)}<div></div>${k(1)}<div></div>${k(2)}<div></div></div>`}
    return`<div class="mono">a = (${fmt(ct.a[0],2)}, ${fmt(ct.a[1],2)}) <span class="muted">∼ 𝒩(μ, σ²I)</span></div>`}})}

/* 4 — policies */
{let kind='greedy',hi=-1;const run=new Runner({pi:()=>model().pi[kind],onStep:(r,st)=>hi=st.a});
scenes.push({controls:()=>[{type:'seg',label:'Policy',opts:[['greedy','Deterministic'],['soft','Stochastic'],['random','Random']],get:()=>kind,set:v=>{kind=v;run.reset()}},...(kind==='soft'?[tempCtl]:[])],
  enter(){run.reset()},update:dt=>run.update(dt,playing),step:()=>run.manual(),reset:()=>run.reset(),
  draw(){drawWorld({pi:model().pi[kind]});drawAgent(run.pos())},
  live(){const s=run.steps.length?run.steps.at(-1).s:run.s;return policyLive(model().pi[kind],s,hi)}})}

/* Stochastic policies in practice */
{let mode='cat',hi=-1;const run=new Runner({pi:()=>model().pi.soft,onStep:(r,st)=>hi=st.a});const ct=makeCont();
scenes.push({controls:()=>[{type:'seg',label:'Policy',opts:[['cat','Categorical'],['gauss','Diagonal Gaussian']],get:()=>mode,set:v=>{mode=v;run.reset();ct.reset()}},
    ...(mode==='cat'?[tempCtl]:[{type:'range',label:'σ',min:.1,max:1.5,step:.05,get:()=>P.sigma,set:v=>P.sigma=v,fmt:v=>v.toFixed(2)}])],
  enter(){run.reset();ct.reset()},update(dt){mode==='cat'?run.update(dt,playing):ct.update(dt,playing)},
  step(){mode==='cat'?run.manual():ct.manual()},reset(){run.reset();ct.reset()},
  draw(){drawWorld();mode==='cat'?drawAgent(run.pos()):ct.draw()},
  live(){if(mode==='cat'){const s=run.steps.length?run.steps.at(-1).s:run.s,q=model().Qs[s],p=model().pi.soft[s];
      return`<table class="t"><tr><th>${lbl(s)}</th>${[0,1,2,3].map(a=>`<th>${ARW[a]}</th>`).join('')}</tr><tr><td class="muted">logit</td>${q.map(v=>`<td>${fmt(v/P.temp,2)}</td>`).join('')}</tr><tr><td class="muted">π(a|s)</td>${p.map((v,a)=>`<td style="${a===hi?'color:var(--hl);font-weight:700':''}">${v.toFixed(2)}</td>`).join('')}</tr></table>`}
    const v2=a=>`(${fmt(a[0],2)}, ${fmt(a[1],2)})`;
    return`<div class="mono">μ = ${v2(ct.mu)} &nbsp; σ = ${P.sigma.toFixed(2)}<br>z = ${v2(ct.z)} → a = <b>${v2(ct.raw)}</b></div>`}})}

/* 5 — trajectories */
{let kind='greedy',trails=[];const run=new Runner({pi:()=>model().pi[kind],onEnd:r=>{trails.push(r.trail.slice());if(trails.length>12)trails.shift()}});
scenes.push({controls:()=>[kindSeg(()=>kind,v=>{kind=v;run.reset();trails=[]},['greedy','soft']),slipCtl],
  onModel(){trails=[]},
  enter(){run.reset();trails=[]},update:dt=>run.update(dt,playing),step:()=>run.manual(),reset(){run.reset();trails=[]},
  draw(){drawWorld({trails,trailCol:tk.muted,trailAlpha:i=>.15+.45*(i+1)/trails.length});drawTrail(run);drawAgent(run.pos())},
  live(){const ch=[`<span>s₀</span>`];run.steps.forEach((st,i)=>{const sl=st.act!==st.a;ch.push(`<span class="a ${sl?'slip':''}">${ARW[st.a]}${sl?'↯':''}</span>`,`<span>${lbl(st.s2)}</span>`)});
    const uniq=new Set(trails.map(t=>t.join())).size;
    return`<div class="chips">${ch.slice(-17).join('')}</div><div class="muted">${trails.length} past trajectories · ${uniq} distinct</div>`}})}

/* 6 — reward & return */
{let mode='disc',hist=[];const ret=steps=>steps.reduce((G,st)=>G+(mode==='disc'?P.gamma**st.t:1)*st.r,0);
const run=new Runner({pi:()=>model().pi.soft,horizon:()=>mode==='fin'?P.horizon:120,
  onStep:(r,st)=>addFloat(st.s2,(st.r>0?'+':'')+fmt(st.r,0),st.r>0?'pos':'neg'),onEnd:r=>{hist.unshift(ret(r.steps));hist=hist.slice(0,5)}});
scenes.push({controls:()=>[{type:'seg',label:'Return',opts:[['fin','Finite horizon'],['disc','Discounted']],get:()=>mode,set:v=>{mode=v;run.reset();hist=[]}},
    mode==='disc'?{...gammaCtl,set:v=>{P.gamma=v;M=null}}:{type:'range',label:'Horizon T',min:3,max:20,step:1,get:()=>P.horizon,set:v=>P.horizon=v,fmt:v=>v}],
  enter(){run.reset();hist=[]},update:dt=>run.update(dt,playing),step:()=>run.manual(),reset(){run.reset();hist=[]},
  draw(dt){drawWorld();drawTrail(run);drawAgent(run.pos());drawFloats(dt)},
  live(){let G=0;const rows=run.steps.map(st=>{const w=mode==='disc'?P.gamma**st.t:1;G+=w*st.r;return`<tr><td>${st.t}</td><td class="${sgn(st.r)}">${fmt(st.r,0)}</td><td>${mode==='disc'?w.toFixed(3):'1'}</td><td>${fmt(G,2)}</td></tr>`}).slice(-4).join('');
    return`<div style="display:flex;gap:36px;align-items:flex-end"><table class="t"><tr><th>t</th><th>r<sub>t</sub></th><th>${mode==='disc'?'γ<sup>t</sup>':'weight'}</th><th>R so far</th></tr>${rows||'<tr><td colspan="4" class="muted">waiting…</td></tr>'}</table>
    <div><div class="muted" style="font-size:15px">R(τ)</div><div class="big ${sgn(G)}">${fmt(G,2)}</div></div></div>
    <div class="muted">Earlier episodes: <span class="mono">${hist.map(h=>fmt(h,2)).join(' · ')||'—'}</span></div>`}})}

/* 7 — the RL problem */
{let kind='random',rets=[],visits=new Array(N).fill(0),last=[];
const reset=()=>{rets=[];visits=new Array(N).fill(0);last=[]};
function rollout(){const pi=model().pi[kind];let s=S0,G=0,d=1,t=0;const tr=[s];while(!isTerm(s)&&t<200){visits[s]++;const{s2}=sampleMove(s,sampleCat(pi[s]));G+=d*reward(s2);d*=P.gamma;s=s2;t++;tr.push(s)}rets.push(G);last=tr}
let acc=0;
scenes.push({controls:()=>[kindSeg(()=>kind,v=>{kind=v;reset()}),gammaCtl,slipCtl],
  onModel:reset,enter:reset,reset,step:rollout,
  update(dt){if(!playing)return;acc+=dt*P.speed*12;while(acc>=1){acc--;rollout()}},
  panelCanvas:true,
  draw(){const vm=Math.max(...visits);drawWorld({visits,vmax:vm,trails:last.length?[last]:[],trailCol:tk.hl,trailAlpha:()=>.9})},
  drawPanel(pg,w,h){pg.clearRect(0,0,w,h);const lo=-60,hi=10,nb=35,bins=new Array(nb).fill(0);for(const r of rets){const b=Math.max(0,Math.min(nb-1,Math.floor((r-lo)/(hi-lo)*nb)));bins[b]++}
    const mx=Math.max(1,...bins),pl=8,pr=8,pt=22,pb=26,bw=(w-pl-pr)/nb,X=v=>pl+(v-lo)/(hi-lo)*(w-pl-pr);
    pg.strokeStyle=tk.line;pg.beginPath();pg.moveTo(pl,h-pb);pg.lineTo(w-pr,h-pb);pg.stroke();
    bins.forEach((c,i)=>{const bh=(h-pt-pb)*c/mx;pg.fillStyle=rgba(tk.agent,.75);pg.fillRect(pl+i*bw+1,h-pb-bh,bw-2,bh)});
    pg.fillStyle=tk.muted;pg.font=font(400,13);pg.textAlign='center';pg.textBaseline='top';for(const v of[-60,-40,-20,0])pg.fillText(fmt(v,0),Math.min(w-pr-10,Math.max(pl+10,X(v))),h-pb+6);
    const J=Vpi(kind)[S0];pg.setLineDash([5,4]);pg.strokeStyle=tk.ink;pg.lineWidth=1.5;pg.beginPath();pg.moveTo(X(J),pt-2);pg.lineTo(X(J),h-pb);pg.stroke();pg.setLineDash([]);
    pg.textBaseline='bottom';pg.fillStyle=tk.ink;pg.fillText('exact J',Math.min(w-30,Math.max(30,X(J))),pt-4);
    if(rets.length){const m=rets.reduce((a,b)=>a+b)/rets.length;pg.strokeStyle=tk.hl;pg.lineWidth=3;pg.beginPath();pg.moveTo(X(m),pt+10);pg.lineTo(X(m),h-pb);pg.stroke()}},
  live(){const n=rets.length,m=n?rets.reduce((a,b)=>a+b)/n:0;return`<div class="stats"><div><div class="muted">episodes</div><div class="big">${n}</div></div><div><div class="muted">average R(τ)</div><div class="big" style="color:var(--hl)">${n?fmt(m,2):'—'}</div></div><div><div class="muted">exact J(π)</div><div class="big">${fmt(Vpi(kind)[S0],2)}</div></div></div>`}})}

/* 8 — value functions */
{let kind='random',show='V',V=new Array(N).fill(0),k=0,delta=0,acc=0;const reset=()=>{V=new Array(N).fill(0);k=0;delta=0};
const sweep=()=>{const o=evalSweep(V,model().pi[kind]);V=o.V;delta=o.delta;k++};
scenes.push({controls:()=>[kindSeg(()=>kind,v=>{kind=v;reset()}),{type:'seg',label:'Show',opts:[['V','V'],['Q','Q']],get:()=>show,set:v=>show=v},gammaCtl,slipCtl],
  onModel:reset,enter:reset,reset,step:sweep,
  update(dt){if(!playing||(k>0&&delta<1e-4))return;acc+=dt*P.speed;if(acc>.45){acc=0;sweep()}},
  draw(){show==='V'?drawWorld({heat:V,nums:V}):drawWorld({q:qFrom(V)})},
  live(){return`<div class="stats"><div><div class="muted">sweep</div><div class="big">${k}</div></div><div><div class="muted">largest change</div><div class="big">${k?delta.toFixed(3):'—'}</div></div><div><div class="muted">V(start)</div><div class="big ${sgn(V[S0])}">${fmt(V[S0],2)}</div></div></div>`}})}

/* 9 — optimal */
{let show='V',V=new Array(N).fill(0),k=0,delta=0,acc=0;const reset=()=>{V=new Array(N).fill(0);k=0;delta=0};
const sweep=()=>{const o=viSweep(V);V=o.V;delta=o.delta;k++};
scenes.push({controls:()=>[{type:'seg',label:'Show',opts:[['V','V* + a*'],['Q','Q*']],get:()=>show,set:v=>show=v},gammaCtl,slipCtl],
  onModel:reset,enter:reset,reset,step:sweep,
  update(dt){if(!playing||(k>0&&delta<1e-4))return;acc+=dt*P.speed;if(acc>.45){acc=0;sweep()}},
  draw(){const Q=qFrom(V);if(show==='V'){const pi=k?ALL.map(s=>{const p=[0,0,0,0];p[argmax(Q[s])]=1;return p}):null;drawWorld({heat:V,nums:V,pi})}else drawWorld({q:Q,qMark:true})},
  live(){return`<div class="stats"><div><div class="muted">sweep</div><div class="big">${k}</div></div><div><div class="muted">largest change</div><div class="big">${k?delta.toFixed(3):'—'}</div></div></div>`}})}

/* 10 — Bellman */
{let mode='pi',sel=17,ai=0,acc=0;
scenes.push({controls:()=>[{type:'seg',label:'Equation',opts:[['pi','V^π (random)'],['star','V*']],get:()=>mode,set:v=>mode=v},gammaCtl,slipCtl],
  click:s=>{sel=s;ai=0},reset:()=>ai=0,step:()=>ai=(ai+1)%4,enter(){},
  update(dt){if(!playing)return;acc+=dt*P.speed;if(acc>1.8){acc=0;ai=(ai+1)%4}},
  draw(){const V=mode==='pi'?Vpi('random'):model().Vs;drawWorld({heat:V,nums:V,numDigits:2,sel});
    const c=center(sel),cs=geo.cs;for(const{s2,p}of trans(sel,ai)){
      if(s2===sel){g.beginPath();g.arc(c.x+DC[ai]*cs*.32,c.y+DR[ai]*cs*.32,cs*.12,0,7);g.strokeStyle=tk.hl;g.lineWidth=3;g.stroke();
        text(p.toFixed(2),c.x+DC[ai]*cs*.32,c.y+DR[ai]*cs*.32,tk.ink,cs*.1);continue}
      const d=center(s2),mx=c.x+(d.x-c.x)*.36,my=c.y+(d.y-c.y)*.36;arrow(c.x,c.y,d.x-(d.x-c.x)*.28,d.y-(d.y-c.y)*.28,2+5*p,tk.hl);
      const tw=cs*.3;rr(mx-tw/2,my-cs*.09,tw,cs*.18,cs*.05);g.fillStyle=tk.surface;g.fill();text(p.toFixed(2),mx,my,tk.ink,cs*.12)}
    text(ARW[ai]+' intended',c.x,c.y-cs*.33,tk.ink,cs*.1,600,'sans')},
  live(){const V=mode==='pi'?Vpi('random'):model().Vs,Q=[0,1,2,3].map(a=>backup(V,sel,a)),g_=P.gamma.toFixed(2);
    const lines=trans(sel,ai).map(({s2,p,r})=>`<div>${p.toFixed(2)} × (${fmt(r,0).padStart(3,' ')} + ${g_}·${(v=>v<0?'('+fmt(v,2)+')':fmt(v,2))(isTerm(s2)?0:V[s2])}) <span class="muted">→ ${lbl(s2)}</span></div>`).join('');
    const tot=mode==='pi'?`V<sup>π</sup>${lbl(sel)} = ¼ Σ Q = ${fmt(Q.reduce((a,b)=>a+b)/4,2)}`:`V*${lbl(sel)} = max Q = ${fmt(Math.max(...Q),2)}`;
    return`<div class="calc"><div class="muted">Q(${lbl(sel)}, ${ARW[ai]}) =</div>${lines}<div>= <b>${fmt(Q[ai],2)}</b></div>
    <div class="qs">${Q.map((q,a)=>`<span class="${a===ai?'on':''}">${ARW[a]} ${fmt(q,2)}</span>`).join('')}</div><div class="tot">${tot}</div></div>`}})}

/* 11 — advantage */
{let kind='random',sel=17;
scenes.push({controls:()=>[kindSeg(()=>kind,v=>kind=v),...(kind==='soft'?[tempCtl]:[]),gammaCtl,slipCtl],
  click:s=>sel=s,enter(){},
  draw(){const V=Vpi(kind),Q=Qpi(kind),A=Q.map((q,s)=>q.map(x=>NT.includes(s)?x-V[s]:0));const sc=Math.max(1,...A.flat().map(Math.abs));drawWorld({q:A,qScale:sc,sel})},
  live(){const V=Vpi(kind)[sel],Q=Qpi(kind)[sel],pi=model().pi[kind][sel],A=Q.map(q=>q-V);
    return`<table class="t"><tr><th>${lbl(sel)}</th>${[0,1,2,3].map(a=>`<th>${ARW[a]}</th>`).join('')}</tr><tr><td class="muted">Q</td>${Q.map(v=>`<td>${fmt(v,2)}</td>`).join('')}</tr><tr><td class="muted">A</td>${A.map(v=>`<td class="${sgn(v)}"><b>${fmt(v,2)}</b></td>`).join('')}</tr></table><div class="muted">V<sup>π</sup> = ${fmt(V,2)} · click another cell</div>`}})}

/* Formalism */
{const run=new Runner({pi:()=>model().pi.soft});
scenes.push({controls:()=>[slipCtl],enter(){run.reset()},update:dt=>run.update(dt,playing),step:()=>run.manual(),reset:()=>run.reset(),
  draw(){drawWorld();drawTrail(run);drawAgent(run.pos())},
  live(){const pi=model().pi.soft;let lp=0;for(const st of run.steps){const pt=trans(st.s,st.a).find(x=>x.s2===st.s2);lp+=Math.log10(pi[st.s][st.a]*(pt?pt.p:0))}
    const ex=Math.floor(lp),man=10**(lp-ex);
    return`<div class="mono">S: 22 cells · A: 4 moves · ρ₀: start ${lbl(S0)} · slip ${P.slip.toFixed(2)}</div><div>P(τ|π) for this episode so far (${run.steps.length} steps): <span class="big">${run.steps.length?`${man.toFixed(2)}×10<sup>${ex}</sup>`:'1'}</span></div>`}})}


/* ===== Part 3 ===== */
scenes.push(header('soft'));

/* Gradient ascent on the policy */
{let L=makePG3('base'),acc=0;const reset=()=>{L=makePG3('base')};
scenes.push({controls:()=>[],enter:reset,reset,onModel:reset,step(){L.trainEpoch()},
  update(dt){if(!playing)return;acc+=dt*P.speed*8;while(acc>=1){acc--;L.trainEpoch()}},
  draw(){drawWorld({pi:L.table()})},
  live(){const th=L.th[S0],p=L.pi(S0),c=L.curve.slice(-10),a=c.length?c.reduce((x,q)=>x+q.y,0)/c.length:null;
    return`<div class="stats"><div><div class="muted">epoch</div><div class="big">${L.epoch}</div></div><div><div class="muted">avg R(τ)</div><div class="big ${a==null?'':sgn(a)}">${a==null?'—':fmt(a,1)}</div></div></div>
    <table class="t"><tr><th>start ${lbl(S0)}</th>${ARW.map(x=>`<th>${x}</th>`).join('')}</tr><tr><td class="muted">θ</td>${th.map(v=>`<td>${fmt(v,2)}</td>`).join('')}</tr><tr><td class="muted">π<sub>θ</sub></td>${p.map(v=>`<td>${v.toFixed(2)}</td>`).join('')}</tr></table>`}})}

/* Derivation, step 1 */
{const run=new Runner({pi:()=>model().pi.soft,horizon:()=>T3});
scenes.push({controls:()=>[],enter(){run.reset()},update:dt=>run.update(dt,playing),step:()=>run.manual(),reset:()=>run.reset(),
  draw(){drawWorld();drawTrail(run);drawAgent(run.pos())},
  live(){const pi=model().pi.soft;let lpi=0,lp=0;const rows=run.steps.map(st=>{const p=pi[st.s][st.a],e=trans(st.s,st.a).find(x=>x.s2===st.s2).p;lpi+=Math.log(p);lp+=Math.log(e);
      return`<tr><td>${st.t}</td><td>${ARW[st.a]}</td><td style="color:var(--agent)">${p.toFixed(2)}</td><td class="muted">${e.toFixed(2)}</td></tr>`}).slice(-4).join('');
    return`<div style="display:flex;gap:30px;align-items:flex-end"><table class="t"><tr><th>t</th><th>a</th><th>π<sub>θ</sub>(a|s)</th><th>P(s′|s,a)</th></tr>${rows||'<tr><td colspan="4" class="muted">waiting…</td></tr>'}</table>
    <div class="mono">Σ log π<sub>θ</sub> = <b style="color:var(--agent)">${fmt(lpi,2)}</b><br>Σ log P = <b class="muted">${fmt(lp,2)}</b></div></div>`}})}

/* Derivation, step 2: estimate from a batch */
{let L,eps=[],shown=0,phase='collect',acc=0,lastAvg=null;
const newBatch=()=>{eps=[];for(let i=0;i<L.batch;i++)eps.push(L.rollout());shown=0;phase='collect'};
const reset=()=>{L=makePG3('R');lastAvg=null;newBatch()};
const tick=()=>{if(phase==='collect'){shown++;if(shown>=eps.length)phase='update'}else{lastAvg=L.update(eps);newBatch()}};
reset();
scenes.push({controls:()=>[],enter:reset,reset,onModel:reset,step:tick,
  update(dt){if(!playing)return;acc+=dt*P.speed;if(acc>=(phase==='collect'?.18:.5)){acc=0;tick()}},
  draw(){drawWorld({pi:L.table()});eps.slice(0,shown).forEach(ep=>{const end=ep.at(-1).s2;drawEpisode(ep,rgba(typ(end)==='G'?tk.pos:typ(end)==='L'?tk.neg:tk.muted,.55))})},
  live(){return`<div class="stats"><div><div class="muted">epoch</div><div class="big">${L.epoch}</div></div><div><div class="muted">batch</div><div class="big">${shown}/${L.batch}</div></div><div><div class="muted">last batch avg R(τ)</div><div class="big ${lastAvg==null?'':sgn(lastAvg)}">${lastAvg==null?'—':fmt(lastAvg,1)}</div></div></div>`}})}

/* EGLP lemma */
{let n=0,sum=[0,0,0,0],last=null,lastA=-1,acc=0,fast=false;
const pi=()=>model().pi.soft[S0];
const reset=()=>{n=0;sum=[0,0,0,0];last=null;lastA=-1};
const sample=()=>{const p=pi(),a=sampleCat(p);last=p.map((q,b)=>(b===a)-q);lastA=a;sum=sum.map((x,b)=>x+last[b]);n++};
function bars(y0,h,vals,title,hiA){const x0=W*.12,w=W*.76,bw=w/4,mid=y0+h/2;
  text(title,x0,y0-12,tk.muted,18,500,'sans','left');g.strokeStyle=tk.line;g.lineWidth=1.5;g.beginPath();g.moveTo(x0,mid);g.lineTo(x0+w,mid);g.stroke();
  for(let a=0;a<4;a++){const v=vals?vals[a]:0,bh=v*h/2,x=x0+a*bw+bw*.2;g.fillStyle=rgba(v>=0?tk.pos:tk.neg,.8);g.fillRect(x,v>=0?mid-bh:mid,bw*.6,Math.abs(bh));
    text(ARW[a],x+bw*.3,y0+h+16,a===hiA?tk.hl:tk.ink,22,600);if(vals)text(fmt(v,2),x+bw*.3,v>=0?mid-bh-14:mid-bh+14,tk.ink,15)}}
scenes.push({controls:()=>[{type:'seg',label:'Animation',opts:[['watch','One sample at a time'],['fast','Fast-forward']],get:()=>fast?'fast':'watch',set:v=>fast=v==='fast'}],
  enter:reset,reset,step:sample,
  update(dt){if(!playing)return;if(fast){for(let i=0;i<Math.round(20*P.speed);i++)sample();return}acc+=dt*P.speed;if(acc>.4){acc=0;sample()}},
  draw(){g.clearRect(0,0,W,H);bars(H*.08,H*.32,last,lastA<0?'latest sample':`latest sample: a = ${ARW[lastA]}`,lastA);bars(H*.56,H*.32,n?sum.map(x=>x/n):null,`average of ${n} samples`,-1)},
  live(){const m=n?sum.map(x=>x/n):[0,0,0,0];return`<div class="stats"><div><div class="muted">samples N</div><div class="big">${n}</div></div><div><div class="muted">size of average</div><div class="big">${n?Math.hypot(...m).toFixed(3):'—'}</div></div></div><div class="mono">π(·|start) = [${pi().map(v=>v.toFixed(2)).join(', ')}]</div>`}})}

/* Episode-weight slides: reward-to-go, baselines */
function epScene(cfg){let lastEp=null,mode=cfg.modes?cfg.modes[0][0]:null;const run=new Runner({pi:()=>model().pi.soft,horizon:()=>T3,hold:2.6,onEnd:r=>lastEp=r.steps.slice()});
  const shown=()=>run.done?run.steps:lastEp;
  return{controls:()=>cfg.modes?[{type:'seg',label:'Weight each action by',opts:cfg.modes,get:()=>mode,set:v=>mode=v}]:[],
    enter(){run.reset();lastEp=null},reset(){run.reset();lastEp=null},update:dt=>run.update(dt,playing),step:()=>run.manual(),
    draw(){drawWorld(cfg.heat?{heat:cfg.heat()}:{});const ep=shown();
      if(ep){drawEpisode(ep,rgba(tk.muted,.5));const w=cfg.weights(ep,mode);ep.forEach((e,i)=>{const a=center(e.s),b=center(e.s2),cs=geo.cs;
        const x=e.s2===e.s?a.x+DC[e.act]*cs*.3:(a.x+b.x)/2,y=e.s2===e.s?a.y+DR[e.act]*cs*.3:(a.y+b.y)/2;pill(x,y,fmt(w[i],1),w[i]>=0?tk.pos:tk.neg)})}
      if(!run.done){drawTrail(run)}drawAgent(run.pos())},
    live(){const ep=shown();if(!ep)return'<div class="muted">Weights appear when the first episode ends. The pills on the grid show each action’s weight.</div>';return cfg.table(ep,mode)}}}
const rowsOf=(ep,f)=>{const idx=ep.length<=8?ep.map((_,i)=>i):[0,1,2,3,-1,ep.length-3,ep.length-2,ep.length-1];return idx.map(i=>i<0?'<tr><td colspan="6" class="muted" style="text-align:center">⋯</td></tr>':f(ep[i],i)).join('')};
scenes.push(epScene({modes:[['R','R(τ)'],['rtg','reward-to-go']],
  weights:(ep,m)=>m==='R'?ep.map(()=>epReturn(ep)):rewardToGo(ep),
  table:(ep,m)=>{const R=epReturn(ep),rtg=rewardToGo(ep),on=c=>c===m?'style="color:var(--hl);font-weight:700"':'';
    return`<table class="t"><tr><th>t</th><th>a<sub>t</sub></th><th>r<sub>t</sub></th><th ${on('R')}>R(τ)</th><th ${on('rtg')}>R̂<sub>t</sub></th></tr>${rowsOf(ep,(e,i)=>`<tr><td>${i}</td><td>${ARW[e.a]}</td><td class="${sgn(e.r)}">${fmt(e.r,0)}</td><td ${on('R')}>${fmt(R,0)}</td><td ${on('rtg')}>${fmt(rtg[i],0)}</td></tr>`)}</table>`}}));
scenes.push(epScene({heat:V1soft,
  weights:ep=>{const V=V1soft(),rtg=rewardToGo(ep);return ep.map((e,i)=>rtg[i]-V[e.s])},
  table:ep=>{const V=V1soft(),rtg=rewardToGo(ep);return`<table class="t"><tr><th>t</th><th>r<sub>t</sub></th><th>R̂<sub>t</sub></th><th>V(s<sub>t</sub>)</th><th>R̂<sub>t</sub> − V</th></tr>${rowsOf(ep,(e,i)=>`<tr><td>${i}</td><td class="${sgn(e.r)}">${fmt(e.r,0)}</td><td>${fmt(rtg[i],0)}</td><td>${fmt(V[e.s],1)}</td><td class="${sgn(rtg[i]-V[e.s])}"><b>${fmt(rtg[i]-V[e.s],1)}</b></td></tr>`)}</table>`}}));

/* Other forms + race */
{const KINDS=[['R','R(τ)',()=>tk.muted],['rtg','reward-to-go',()=>tk.agent],['base','reward-to-go − baseline',()=>tk.hl]],SEEDS=3,XMAX=200;let Ls,acc=0;
const reset=()=>{Ls=KINDS.map(([k])=>Array.from({length:SEEDS},()=>makePG3(k)))};
const avgCurve=arr=>{const n=Math.min(...arr.map(l=>l.curve.length));return Array.from({length:n},(_,i)=>({x:i+1,y:arr.reduce((s,l)=>s+l.curve[i].y,0)/arr.length}))};
reset();
scenes.push({controls:()=>[],enter:reset,reset,onModel:reset,
  step(){if(Ls[0][0].epoch<XMAX)Ls.flat().forEach(l=>l.trainEpoch())},
  update(dt){if(!playing)return;acc+=dt*P.speed*15;while(acc>=1&&Ls[0][0].epoch<XMAX){acc--;Ls.flat().forEach(l=>l.trainEpoch())}if(Ls[0][0].epoch>=XMAX)acc=0},
  draw(){g.clearRect(0,0,W,H);chart(g,W,H,KINDS.map(([k,lab,col],i)=>({c:avgCurve(Ls[i]),col:col(),label:lab})),{xMax:XMAX,lo:-60,hi:10,ticks:[-60,-40,-20,0],xlabel:'epochs (10 episodes each)',ref:V1star()[S0],win:10,fs:17,lw:3.5,pr:20})},
  live(){const f=i=>{const c=avgCurve(Ls[i]).slice(-10);return c.length?fmt(c.reduce((s,p)=>s+p.y,0)/c.length,1):'—'};
    return`<div class="stats">${KINDS.map(([k,lab,col],i)=>`<div><div class="muted">${lab}</div><div class="big" style="color:${col()}">${f(i)}</div></div>`).join('')}</div><div class="muted">epoch ${Ls[0][0].epoch} of ${XMAX}</div>`}})}

/* Closing */
{const run=new Runner({pi:()=>model().pi.greedy});
scenes.push({controls:()=>[],enter(){run.reset()},update:dt=>run.update(dt,playing),step:()=>run.manual(),reset:()=>run.reset(),
  draw(){const m=model();drawWorld({heat:m.Vs,pi:m.pi.greedy});drawTrail(run);drawAgent(run.pos())}})}
/* ================= Registry ================= */
const SCENE_IDS=['cover','part1','what-can-rl-do','meet','loop','states','actions','policies','stochastic','trajectories','return','problem',
  'values','optimal','bellman','advantage','formal','part3','ascent','ingredients','result','eglp','reward-to-go','baselines','forms','closing'];
if(scenes.length!==SCENE_IDS.length)throw new Error(`gridworld: ${scenes.length} scenes but ${SCENE_IDS.length} ids`);
const byId=Object.fromEntries(SCENE_IDS.map((id,i)=>[id,scenes[i]]));
export function getScene(id){const s=byId[id];if(!s)throw new Error('Unknown scene: '+id);return s}
let activeKey=0;
export function activate(sc,seedKey){active=sc;activeKey=seedKey;floats=[];reseed(seedKey);sc.enter&&sc.enter()}
export function resetActive(){if(!active)return;floats=[];reseed(activeKey);active.reset&&active.reset()}
export function stepActive(){active&&active.step&&active.step()}
export const isActive=sc=>active===sc;
export {P,R,C,isWall,isTerm};
