#!/bin/sh
grep "dodag" out-$1.cap > dodag-$1.cap
grep "rank:" out-$1.cap > rpl-$1.cap
grep "scope: global" out-$1.cap > addr-$1.cap
grep "from" out-$1.cap > trace-$1.cap
