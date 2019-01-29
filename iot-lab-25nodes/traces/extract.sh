#!/bin/sh
grep "rank:" out-$1.cap > rpl-$1.cap
grep "scope: global" out-$1.cap > addr-$1.cap
