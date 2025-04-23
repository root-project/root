#pragma once

#include "TMap.h"

class UserClass {};

class UserClassViaTypedef{};

typedef UserClassViaTypedef UserClass_t;

class UserClassViaUsing{};

using UserClass_t2 = UserClassViaUsing;

class UserClassNotSelected {};
using UserClassOnlyTypedef_t = UserClassNotSelected;
