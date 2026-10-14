//! A fake ASIO driver for the tests: the real method table (see `sys`), backed by Rust,
//! with a clock thread that calls `bufferSwitch` like hardware does and records what the
//! host wrote into each half buffer.

use super::host::{AsioError, Driver};
use super::sys::{
    self, BufferInfo, Callbacks, ChannelInfo, IAsio, IAsioVtbl, Int64Parts, IoFormat,
};
use super::DriverSource;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// What the fake driver offers.
#[derive(Clone, Debug)]
pub struct Spec {
    pub outputs: i32,
    pub pcm_type: i32,
    /// `None`: no DSD mode.
    pub dsd_type: Option<i32>,
    pub rates: Vec<f64>,
    pub dsd_rates: Vec<f64>,
    pub buffer: i32,
    pub output_ready: bool,
    pub init_ok: bool,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            outputs: 2,
            pcm_type: sys::ST_INT32_LSB,
            dsd_type: Some(sys::ST_DSD_INT8_MSB1),
            rates: vec![
                44_100.0, 48_000.0, 88_200.0, 96_000.0, 176_400.0, 192_000.0, 352_800.0, 384_000.0,
            ],
            dsd_rates: vec![2_822_400.0, 5_644_800.0, 11_289_600.0],
            buffer: 256,
            output_ready: true,
            init_ok: true,
        }
    }
}

/// Mutable driver state.
#[derive(Default)]
pub struct State {
    pub dsd: bool,
    pub rate: f64,
    pub buffers: Vec<[Vec<u8>; 2]>,
    pub callbacks: Option<*mut Callbacks>,
    pub started: bool,
    pub output_ready_calls: u64,
    /// Every half buffer delivered, per channel, concatenated.
    pub played: Vec<Vec<u8>>,
    pub size: i32,
    pub refs: i32,
}

#[repr(C)]
pub struct Fake {
    vtbl: *const IAsioVtbl,
    pub spec: Spec,
    pub state: Mutex<State>,
    running: AtomicBool,
}

unsafe impl Send for Fake {}
unsafe impl Sync for Fake {}

fn me<'a>(this: *mut IAsio) -> &'a Fake {
    // SAFETY: every `IAsio` handed out by this module is a `Fake`.
    unsafe { &*(this as *const Fake) }
}

unsafe extern "system" fn query_interface(
    _: *mut IAsio,
    _: *const c_void,
    _: *mut *mut c_void,
) -> i32 {
    -1
}
unsafe extern "system" fn add_ref(this: *mut IAsio) -> u32 {
    let mut s = me(this).state.lock().unwrap();
    s.refs += 1;
    s.refs as u32
}
unsafe extern "system" fn release(this: *mut IAsio) -> u32 {
    let mut s = me(this).state.lock().unwrap();
    s.refs -= 1;
    s.refs as u32
}
unsafe extern "system" fn init(this: *mut IAsio, _: *mut c_void) -> i32 {
    me(this).spec.init_ok as i32
}
unsafe extern "system" fn get_driver_name(_: *mut IAsio, name: *mut u8) {
    let n = b"Fake ASIO\0";
    unsafe { std::ptr::copy_nonoverlapping(n.as_ptr(), name, n.len()) };
}
unsafe extern "system" fn get_driver_version(_: *mut IAsio) -> i32 {
    1
}
unsafe extern "system" fn get_error_message(_: *mut IAsio, msg: *mut u8) {
    let n = b"no hardware\0";
    unsafe { std::ptr::copy_nonoverlapping(n.as_ptr(), msg, n.len()) };
}
unsafe extern "system" fn start(this: *mut IAsio) -> i32 {
    let f = me(this);
    let mut s = f.state.lock().unwrap();
    if s.callbacks.is_none() {
        return sys::ASE_INVALID_MODE;
    }
    s.started = true;
    sys::ASE_OK
}
unsafe extern "system" fn stop(this: *mut IAsio) -> i32 {
    me(this).state.lock().unwrap().started = false;
    sys::ASE_OK
}
unsafe extern "system" fn get_channels(this: *mut IAsio, i: *mut i32, o: *mut i32) -> i32 {
    unsafe {
        *i = 0;
        *o = me(this).spec.outputs;
    }
    sys::ASE_OK
}
unsafe extern "system" fn get_latencies(this: *mut IAsio, i: *mut i32, o: *mut i32) -> i32 {
    unsafe {
        *i = 0;
        *o = me(this).spec.buffer;
    }
    sys::ASE_OK
}
unsafe extern "system" fn get_buffer_size(
    this: *mut IAsio,
    min: *mut i32,
    max: *mut i32,
    pref: *mut i32,
    gran: *mut i32,
) -> i32 {
    let b = me(this).spec.buffer;
    unsafe {
        *min = b;
        *max = b;
        *pref = b;
        *gran = 0;
    }
    sys::ASE_OK
}
fn rate_ok(f: &Fake, dsd: bool, r: f64) -> bool {
    if dsd {
        f.spec.dsd_rates.contains(&r)
    } else {
        f.spec.rates.contains(&r)
    }
}
unsafe extern "system" fn can_sample_rate(this: *mut IAsio, r: f64) -> i32 {
    let f = me(this);
    let dsd = f.state.lock().unwrap().dsd;
    if rate_ok(f, dsd, r) {
        sys::ASE_OK
    } else {
        sys::ASE_NO_CLOCK
    }
}
unsafe extern "system" fn get_sample_rate(this: *mut IAsio, r: *mut f64) -> i32 {
    unsafe { *r = me(this).state.lock().unwrap().rate };
    sys::ASE_OK
}
unsafe extern "system" fn set_sample_rate(this: *mut IAsio, r: f64) -> i32 {
    let f = me(this);
    let mut s = f.state.lock().unwrap();
    if !rate_ok(f, s.dsd, r) {
        return sys::ASE_NO_CLOCK;
    }
    s.rate = r;
    sys::ASE_OK
}
unsafe extern "system" fn get_clock_sources(_: *mut IAsio, _: *mut c_void, n: *mut i32) -> i32 {
    unsafe { *n = 0 };
    sys::ASE_OK
}
unsafe extern "system" fn set_clock_source(_: *mut IAsio, _: i32) -> i32 {
    sys::ASE_OK
}
unsafe extern "system" fn get_sample_position(
    _: *mut IAsio,
    _: *mut Int64Parts,
    _: *mut Int64Parts,
) -> i32 {
    sys::ASE_SP_NOT_ADVANCING
}
unsafe extern "system" fn get_channel_info(this: *mut IAsio, info: *mut ChannelInfo) -> i32 {
    let f = me(this);
    let info = unsafe { &mut *info };
    if info.channel < 0 || info.channel >= f.spec.outputs || info.is_input != 0 {
        return sys::ASE_INVALID_PARAMETER;
    }
    let dsd = f.state.lock().unwrap().dsd;
    info.is_active = 1;
    info.sample_type = if dsd {
        f.spec.dsd_type.unwrap_or(-1)
    } else {
        f.spec.pcm_type
    };
    sys::ASE_OK
}
fn bytes_of(t: i32) -> usize {
    match t {
        sys::ST_INT16_LSB | sys::ST_INT16_MSB => 2,
        sys::ST_INT24_LSB | sys::ST_INT24_MSB => 3,
        sys::ST_FLOAT64_LSB | sys::ST_FLOAT64_MSB => 8,
        sys::ST_DSD_INT8_MSB1 | sys::ST_DSD_INT8_LSB1 => 1,
        _ => 4,
    }
}
unsafe extern "system" fn create_buffers(
    this: *mut IAsio,
    infos: *mut BufferInfo,
    n: i32,
    size: i32,
    cb: *mut Callbacks,
) -> i32 {
    let f = me(this);
    let mut s = f.state.lock().unwrap();
    let t = if s.dsd {
        f.spec.dsd_type.unwrap_or(-1)
    } else {
        f.spec.pcm_type
    };
    let bytes = if s.dsd {
        size as usize / 8
    } else {
        size as usize * bytes_of(t)
    };
    let infos = unsafe { std::slice::from_raw_parts_mut(infos, n as usize) };
    s.buffers = infos
        .iter()
        .map(|_| [vec![0u8; bytes], vec![0u8; bytes]])
        .collect();
    s.played = vec![Vec::new(); infos.len()];
    for (i, b) in infos.iter_mut().zip(s.buffers.iter_mut()) {
        i.buffers = [
            b[0].as_mut_ptr() as *mut c_void,
            b[1].as_mut_ptr() as *mut c_void,
        ];
    }
    s.callbacks = Some(cb);
    s.size = size;
    sys::ASE_OK
}
unsafe extern "system" fn dispose_buffers(this: *mut IAsio) -> i32 {
    let mut s = me(this).state.lock().unwrap();
    s.callbacks = None;
    sys::ASE_OK
}
unsafe extern "system" fn control_panel(_: *mut IAsio) -> i32 {
    sys::ASE_OK
}
unsafe extern "system" fn future(this: *mut IAsio, selector: i32, opt: *mut c_void) -> i32 {
    let f = me(this);
    let fmt = unsafe { &mut *(opt as *mut IoFormat) };
    let want = fmt.format_type;
    let can =
        want == sys::IO_FORMAT_PCM || (want == sys::IO_FORMAT_DSD && f.spec.dsd_type.is_some());
    match selector {
        sys::FUTURE_CAN_DO_IO_FORMAT if can => sys::ASE_SUCCESS,
        sys::FUTURE_SET_IO_FORMAT if can => {
            f.state.lock().unwrap().dsd = want == sys::IO_FORMAT_DSD;
            sys::ASE_SUCCESS
        }
        _ => sys::ASE_NOT_PRESENT,
    }
}
unsafe extern "system" fn output_ready(this: *mut IAsio) -> i32 {
    let f = me(this);
    if !f.spec.output_ready {
        return sys::ASE_NOT_PRESENT;
    }
    f.state.lock().unwrap().output_ready_calls += 1;
    sys::ASE_OK
}

static VTBL: IAsioVtbl = IAsioVtbl {
    query_interface,
    add_ref,
    release,
    init,
    get_driver_name,
    get_driver_version,
    get_error_message,
    start,
    stop,
    get_channels,
    get_latencies,
    get_buffer_size,
    can_sample_rate,
    get_sample_rate,
    set_sample_rate,
    get_clock_sources,
    set_clock_source,
    get_sample_position,
    get_channel_info,
    create_buffers,
    dispose_buffers,
    control_panel,
    future,
    output_ready,
};

impl Fake {
    pub fn new(spec: Spec) -> Arc<Self> {
        Arc::new(Self {
            vtbl: &VTBL,
            spec,
            state: Mutex::new(State::default()),
            running: AtomicBool::new(false),
        })
    }

    /// One hardware period: the driver hands half `index` to the host, then "plays" it.
    pub fn tick(&self, index: usize) {
        let cb = {
            let s = self.state.lock().unwrap();
            if !s.started {
                return;
            }
            s.callbacks
        };
        let Some(cb) = cb else { return };
        // SAFETY: the callback table is the host's static.
        unsafe { ((*cb).buffer_switch)(index as i32, 1) };
        let mut s = self.state.lock().unwrap();
        let halves: Vec<Vec<u8>> = s.buffers.iter().map(|b| b[index].clone()).collect();
        for (p, h) in s.played.iter_mut().zip(halves) {
            p.extend(h);
        }
    }

    /// Run the clock on a thread (one tick every `period`) until [`Fake::halt`].
    pub fn run(self: &Arc<Self>, period: Duration) -> std::thread::JoinHandle<()> {
        self.running.store(true, Ordering::Release);
        let me = Arc::clone(self);
        std::thread::spawn(move || {
            let mut index = 0;
            while me.running.load(Ordering::Acquire) {
                me.tick(index);
                index ^= 1;
                std::thread::sleep(period);
            }
        })
    }

    pub fn halt(&self) {
        self.running.store(false, Ordering::Release);
    }

    pub fn played(&self) -> Vec<Vec<u8>> {
        self.state.lock().unwrap().played.clone()
    }

    /// Simulate the driver's reset request through the host's message callback.
    pub fn request_reset(&self) {
        let cb = self.state.lock().unwrap().callbacks;
        if let Some(cb) = cb {
            unsafe {
                ((*cb).asio_message)(
                    sys::MSG_RESET_REQUEST,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            };
        }
    }
}

/// A driver source with fake drivers by name.
pub struct FakeSource(pub Vec<(String, Arc<Fake>)>);

impl DriverSource for FakeSource {
    fn names(&self) -> Vec<String> {
        self.0.iter().map(|(n, _)| n.clone()).collect()
    }

    fn load(&self, name: &str) -> Result<Driver, AsioError> {
        let (_, f) = self
            .0
            .iter()
            .find(|(n, _)| n == name)
            .ok_or_else(|| AsioError::host("load", "no such driver"))?;
        let ptr = Arc::as_ptr(f) as *mut IAsio;
        f.state.lock().unwrap().refs += 1;
        // SAFETY: `ptr` is a live `Fake` (kept alive by the source) whose first field is the
        // method table; the reference we just counted is handed over.
        Ok(unsafe { Driver::from_raw(ptr, name) }.expect("non-null"))
    }
}
